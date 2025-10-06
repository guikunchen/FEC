import copy
from typing import Tuple, Optional, Type, Sequence, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.common_types import _size_2_t, _ratio_2_t
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_model
from timm.layers import DropPath, trunc_normal_, to_2tuple
from torch_scatter import scatter_sum

try:
    import math
    from collections import OrderedDict
    from mmcv.runner import _load_checkpoint
    from mmcv.cnn import constant_init, trunc_normal_init, normal_init
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger as det_get_root_logger
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

try:
    if not has_mmdet:
        import math
        from collections import OrderedDict
        from mmcv.runner import _load_checkpoint
        from mmcv.cnn import constant_init, trunc_normal_init, normal_init
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger as seg_get_root_logger
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False


DEFAULT_CFG = {
    'url': '', 'num_classes': 1000, 'classifier': 'head',
    'input_size': (3, 224, 224), 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    'crop_pct': .9, 'interpolation': 'bicubic',
}

CIFAR100_CFG = {
    'url': '', 'num_classes': 100, 'classifier': 'head',
    'input_size': (3, 224, 224), 'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761),
    'crop_pct': .9, 'interpolation': 'bicubic',
}


class LayerNorm(nn.Module):
    """Layer Normalization.

    Come from `A ConvNet for the 2020s`(https://arxiv.org/abs/2201.03545, https://github.com/facebookresearch/ConvNeXt).

    Args:
        normalized_shape (int): The number of expected features in input.
        eps (float): a value added to the denominator for numerical stability. Default: 1e-6.
        data_format (str): The format of the input tensor. If
            `channels_first`, the shape of the input tensor should be
            (B, C, H, W). If `channels_last`, the shape of the input
            tensor should be (B, H, W, C). Default: `channels_first`.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = [normalized_shape, ]
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f'{self.data_format} is unsupported.'
                                      f'Only support "channels_last" and "channels_first".')

    def forward(self, x: torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.bias is not None:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None, None] * x
            return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Args:
        dim (int): Number of input channels.
        hidden_dim (int, optional): Number of hidden channels. Default: None.
        out_dim (int, optional): Number of output channels. Default: None.
        act_layer (Type[nn.Module]): Activation layer. Default: nn.GELU.
        drop_rate (float | tuple): Dropout rate. Default: 0.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, out_dim: Optional[int] = None,
                 act_layer: Type[nn.Module] = nn.GELU, drop_rate: _ratio_2_t = 0.):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        drop_rate = to_2tuple(drop_rate)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop1 = nn.Dropout(drop_rate[0]) if drop_rate[0] > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate[1]) if drop_rate[1] > 0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x.permute(0, 3, 1, 2)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PointAggregation(nn.Module):
    """Point Aggregation.

    Args:
        in_channels (int): Number of input channels.
        embed_dim (int): Number of output channels.
        patch_size (int | tuple): Patch size. Default: 3.
        stride (int | tuple): Stride. Default: 2.
        padding (int | tuple): Padding. Default: 1.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: _size_2_t = 4,
                 stride: _size_2_t = 4, padding: _size_2_t = 0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.aggregate = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = LayerNorm(embed_dim, data_format='channels_first')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aggregate(x)
        x = self.norm(x)
        return x


class ClusterPool(nn.Module):
    """Feature clustering pooling to downsample.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        skip_connection (bool): Whether to use skip connection. Default: True.
    """

    def __init__(self, dim: int, out_dim: int, skip_connection: bool = True):
        super().__init__()
        self.proj_xv = nn.Conv2d(dim, 2 * out_dim, 1)  # clusters, values
        self.clusters = nn.AvgPool2d(2)  # <=> AdaptiveAvgPool2d, but faster
        # self.proj = nn.Conv2d(out_dim, out_dim, 1)

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.down_conv = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = LayerNorm(out_dim, data_format="channels_first")

    def forward(self, x: torch.Tensor):
        agg = self.aggregate_feature(x)
        if self.skip_connection:
            x = self.down_conv(x)
        return self.norm(x + agg) if self.skip_connection else self.norm(agg)

    def aggregate_feature(self, x: torch.Tensor):
        B, C, H, W = x.shape
        xv = self.proj_xv(x).view(B, 2, -1, H, W).permute(1, 0, 2, 3, 4)
        x, v = xv[0], xv[1]  # x, v.shape: [B, C2, H, W]
        C2 = x.shape[1]

        clusters = self.clusters(x).view(B, C2, -1)  # [B, C2, M]
        v_clusters = self.clusters(v).view(B, C2, -1)  # [B, C2, M]
        v = v.view(B, C2, -1)  # [B, C2, N]
        M, N = clusters.shape[2], v.shape[2]  # set M = clusters_h * clusters_w, set N = H * W

        similarity = self.pairwise_cos_sim(clusters.permute(0, 2, 1), x.view(B, C2, -1))  # [B, M, N]
        sim_max_idx = similarity.argmax(dim=1, keepdim=True)  # [B, 1, N]
        mask = torch.zeros_like(similarity, device=similarity.device).scatter_(1, sim_max_idx, 1.)

        v = v.permute(0, 2, 1).reshape(B * N, -1)
        sim_max_idx =  sim_max_idx.view(-1)
        idx_offset = (torch.arange(B, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
        sim_max_idx = sim_max_idx + idx_offset

        agg = scatter_sum(v, sim_max_idx, dim=0, dim_size=B * M).view(B, M, -1).permute(0, 2, 1)
        agg = (agg + v_clusters) / (mask.sum(dim=-1).unsqueeze(1) + 1.0)  # +1.0 to avoid division by zero
        agg = agg.view(B, C2, H // 2, W // 2)
        return agg

    @staticmethod
    def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate the cosine similarity between two tensors.

        Args:
            x1: tensor with shape [B, M, D]
            x2: tensor with shape [B, D, N]

        Returns:
            similarity matrix with shape [B, M, N]
        """

        return F.normalize(x1, dim=2) @ F.normalize(x2, dim=1)


class Cluster(nn.Module):
    """Feature clustering module.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        num_heads (int): Number of heads in context cluster. Default: 4.
        head_dim (int): Dimension of each head in context cluster. Default: 24.
        clusters_hw (tuple): Number of clusters (H W). Default: (2, 2).
        proposals_hw (tuple, optional): Number of cluster proposals (h w). Default: (2, 2).
        fold_factor (tuple): The fold factor of the feature map (feature map -> windows) (H W). Default: None.
        local_compute (bool): Whether to use local computation. Default: False.
    """

    def __init__(self, dim: int, out_dim: int, num_heads: int = 4, head_dim: int = 24,
                 clusters_hw: Tuple[int, int] = (2, 2), proposals_hw: Optional[Tuple[int, int]] = (2, 2),
                 fold_factor: Optional[Tuple[int, int]] = None, local_compute: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.clusters_hw = clusters_hw
        self.num_clusters = clusters_hw[0] * clusters_hw[1]
        if fold_factor is not None:
            self.fold_h, self.fold_w = fold_factor
        else:
            self.fold_h, self.fold_w = to_2tuple(clusters_hw)
        self.local_compute = local_compute and self.fold_h > 1 and self.fold_w > 1

        if self.fold_h > 1 and self.fold_w > 1 and self.local_compute:
            assert self.fold_h & (self.fold_h - 1) == 0 and self.fold_w & (self.fold_w - 1) == 0, \
                f'fold_h and fold_w should be power of 2, but got {self.fold_h} and {self.fold_w}.'  # TODO optional
            self.proposals_hw = proposals_hw
            self.num_proposals = proposals_hw[0] * proposals_hw[1]  # num of proposals in each window
            self.proj_xyv = nn.Conv2d(dim, 3 * num_heads * head_dim, 1)  # clusters, proposals, values
            self.cluster_proposals = nn.AdaptiveAvgPool2d(proposals_hw)

            self.sim_alpha1 = nn.Parameter(torch.zeros(self.num_clusters))
            self.sim_beta1 = nn.Parameter(torch.zeros(self.num_clusters))
            self.sim_alpha2 = nn.Parameter(torch.zeros(self.num_proposals))
            self.sim_beta2 = nn.Parameter(torch.zeros(self.num_proposals))
        else:
            self.proj_xv = nn.Conv2d(dim, 2 * num_heads * head_dim, 1)  # clusters, values

            self.sim_alpha = nn.Parameter(torch.zeros(self.num_clusters))
            self.sim_beta = nn.Parameter(torch.zeros(self.num_clusters))

        self.proj = nn.Conv2d(num_heads * head_dim, out_dim, 1)
        self.clusters = nn.AdaptiveAvgPool2d(clusters_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_compute:
            return self.local_aggregate_distribute(x)
        else:
            return self.aggregate_distribute(x)

    def local_aggregate_distribute(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate the feature maps to the clusters and distribute the clusters to proposals,
        and then distribute the proposals to points. Top-Down implementation: clusters -> proposals -> points.

        Args:
            x (torch.Tensor): The input feature maps with shape [B, C, H, W].

        Returns:
            x (torch.Tensor): The output feature maps with shape [B, C, H, W] after aggregation and distribution.
        """

        # step1: aggregate each proposal into clusters
        B, _, H, W = x.shape
        xyv = self.proj_xyv(x).view(B, 3, self.num_heads * self.head_dim, H, W).permute(1, 0, 2, 3, 4)
        x, y, v = xyv[0], xyv[1], xyv[2]  # x, y, v.shape: [B, num_heads * head_dim, H, W]

        if self.num_heads > 1:  # [B * num_heads, head_dim, H, W]
            x = x.reshape(B * self.num_heads, self.head_dim, H, W)
            y = y.reshape(B * self.num_heads, self.head_dim, H, W)
            v = v.reshape(B * self.num_heads, self.head_dim, H, W)
        B1 = x.shape[0]  # B * num_heads

        x = self.window_partition(x)  # [B1 * fold_h * fold_w, head_dim, Wh, Ww], set Wh = H // fold_h, Ww = W // fold_w
        fold_v = self.window_partition(v)  # [B1 * fold_h * fold_w, head_dim, Wh, Ww]
        B2, _, Wh, Ww = x.shape

        # set M1 = num_proposals * Wh * Ww (unfold), M2 = num_clusters (unfold), M3 = num_proposals (fold)
        proposals = self.cluster_proposals(x)  # [B2, head_dim, proposal_h, proposal_w]
        clusters = self.clusters(y).view(B1, self.head_dim, -1).permute(0, 2, 1)  # [B1, M2, head_dim]
        v_proposals = self.cluster_proposals(fold_v)  # [B2, head_dim, proposal_h, proposal_w]
        v_clusters = self.clusters(v).view(B1, self.head_dim, -1)  # [B1, head_dim, M2]

        similarity = F.softmax(
            (torch.exp(self.sim_alpha1).view(1, -1, 1) * self.pairwise_cos_sim(
                clusters, self.window_reverse(proposals).view(B1, self.head_dim, -1))
             + self.sim_beta1.view(1, -1, 1)), dim=1)  # [B1, M2, M1]

        agg = (self.window_reverse(v_proposals).view(B1, self.head_dim, -1).unsqueeze(2) *
               similarity.unsqueeze(1)).sum(dim=-1)  # agg_clusters, [B1, head_dim, M2]
        agg = (agg + v_clusters) / similarity.sum(dim=-1).unsqueeze(1)

        # step2: distribute clusters to each proposal, shape: [B1, head_dim, M1]
        if self.training:  # avoid the loss becoming nan
            proposals_ = (agg.unsqueeze(3) * similarity.unsqueeze(1)).sum(dim=2)
        else:  # faster training/inference and the acc will be a little higher, but the loss may be nan
            proposals_ = agg @ similarity
        proposals_ = self.window_partition(proposals_.view(B1, self.head_dim, self.fold_h * self.proposals_hw[0],
                                                           self.fold_w * self.proposals_hw[1]))
        proposals = (proposals + proposals_).view(B2, self.head_dim, -1).permute(0, 2, 1)  # [B2, M3, head_dim]

        # step3: aggregate each point into proposals
        similarity = F.softmax(
            (torch.exp(self.sim_alpha2).view(1, -1, 1) * self.pairwise_cos_sim(
                proposals, x.view(B2, self.head_dim, -1))
             + self.sim_beta2.view(1, -1, 1)), dim=1)  # [B2, M3, N], set N = Wh * Ww (fold)

        fold_v = fold_v.view(B2, self.head_dim, -1)  # [B2, head_dim, N]
        agg = (fold_v.unsqueeze(2) * similarity.unsqueeze(1)).sum(dim=-1)  # [B2, head_dim, M3]
        agg = (agg + v_proposals.view(B2, self.head_dim, -1)) / similarity.sum(dim=-1).unsqueeze(1)

        # step4: distribute proposals to each point, shape: [B2, head_dim, N]
        if self.training:
            x = ((agg.unsqueeze(3) * similarity.unsqueeze(1)).sum(dim=2)).view(B2, self.head_dim, Wh, Ww)
        else:
            x = (agg @ similarity).view(B2, self.head_dim, Wh, Ww)
        x = self.window_reverse(x)  # [B1, head_dim, H, W]

        if self.num_heads > 1:
            x = x.view(B, self.num_heads * self.head_dim, H, W)
        return self.proj(x)

    def aggregate_distribute(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate the feature maps to the clusters and distribute the clusters to points.
        Implementation: clusters -> points.

        Args:
            x (torch.Tensor): The input feature maps with shape [B, C, H, W].

        Returns:
            x (torch.Tensor): The output feature maps with shape [B, C, H, W] after aggregation and distribution.
        """

        # step1: aggregate each point into clusters
        B, _, H, W = x.shape
        xv = self.proj_xv(x).view(B, 2, self.num_heads * self.head_dim, H, W).permute(1, 0, 2, 3, 4)
        x, v = xv[0], xv[1]  # x, v.shape: [B, num_heads * head_dim, H, W]
        if self.num_heads > 1:  # [B * num_heads, head_dim, H, W]
            x = x.reshape(B * self.num_heads, self.head_dim, H, W)
            v = v.reshape(B * self.num_heads, self.head_dim, H, W)
        B2 = x.shape[0]

        clusters = self.clusters(x).view(B2, self.head_dim, -1)  # [B2, head_dim, M], set M = clusters_h * clusters_w
        v_clusters = self.clusters(v).view(B2, self.head_dim, -1)  # [B2, head_dim, M]
        v = v.view(B2, self.head_dim, -1)  # [B2, head_dim, N]

        similarity = F.softmax(
            (torch.exp(self.sim_alpha).view(1, -1, 1) * self.pairwise_cos_sim(
                clusters.permute(0, 2, 1),
                x.view(B2, self.head_dim, -1))
             + self.sim_beta.view(1, -1, 1)), dim=1)  # [B2, M, N]

        agg = (v.unsqueeze(2) * similarity.unsqueeze(1)).sum(dim=-1)  # [B2, head_dim, M]
        agg = (agg + v_clusters) / similarity.sum(dim=-1).unsqueeze(1)

        # step2: distribute clusters to each point
        if self.training:
            x = (agg.unsqueeze(3) * similarity.unsqueeze(1)).sum(dim=2)  # [B2, head_dim, N]
        else:
            x = agg @ similarity  # [B2, head_dim, N]

        if self.num_heads > 1:
            x = x.view(B, self.num_heads * self.head_dim, H, W)
        return self.proj(x)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Split the large feature map to small windows.

        Args:
            x (torch.Tensor): The input feature maps with shape [B, C, H, W].

        Returns:
            windows (torch.Tensor): The split windows with shape [B * fold_h * fold_w, C, H // fold_h, W // fold_w].
        """

        B, C, H, W = x.shape
        x = x.view(B, C, self.fold_h, H // self.fold_h, self.fold_w, W // self.fold_w)
        windows = x.permute(0, 2, 4, 1, 3, 5).reshape(
            B * self.fold_h * self.fold_w, C, H // self.fold_h, W // self.fold_w)
        return windows

    def window_reverse(self, windows: torch.Tensor) -> torch.Tensor:
        """Reverse the split windows back to the large feature map.

        Args:
            windows (torch.Tensor): The split windows with shape [B * fold_h * fold_w, C, window_height, window_width].

        Returns:
            x (torch.Tensor): The reversed large feature map with shape [B, C, H, W].
        """

        b, c, Wh, Ww = windows.shape
        B = b // (self.fold_h * self.fold_w)
        x = windows.view(B, self.fold_h, self.fold_w, c, Wh, Ww)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, c, Wh * self.fold_h, Ww * self.fold_w)
        return x

    @staticmethod
    def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate the cosine similarity between two tensors.

        Args:
            x1: tensor with shape [B, M, D]
            x2: tensor with shape [B, D, N]

        Returns:
            similarity matrix with shape [B, M, N]
        """

        return F.normalize(x1, dim=2) @ F.normalize(x2, dim=1)


class ClusterBlock(nn.Module):
    """Cluster Block.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        num_heads (int): Number of heads in context cluster. Default: 4.
        head_dim (int): Dimension of each head in context cluster. Default: 24.
        clusters_hw (tuple): Number of clusters (H W). Default: (2, 2).
        proposals_hw (tuple, optional): Number of cluster proposals (h w). Default: (2, 2).
        fold_factor (tuple): The fold factor of the feature map (feature map -> windows) (H W). Default: None.
        local_compute (bool): Whether to use local computation. Default: False.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        mlp_act_layer (Type[nn.Module]): Hidden activation layer in MLP. Default: nn.GELU.
        mlp_drop_rate (float | tuple): Dropout rate of MLP. Default: 0.
        drop_path_rate (float | tuple): Drop path rate. Default: 0.
        layer_scale (float): The scale of the layer. Default: 1e-5.
        add_identity (bool): Whether to add the identity tensor. Default: True.
        with_cp (bool): Whether to use checkpoint. Default: False.
    """

    def __init__(self, dim: int, out_dim: int, num_heads: int = 4, head_dim: int = 24,
                 clusters_hw: Tuple[int, int] = (2, 2), proposals_hw: Tuple[int, int] = (2, 2),
                 fold_factor: Tuple[int, int] = (1, 1), local_compute: bool = False,
                 mlp_ratio=4., mlp_act_layer: Type[nn.Module] = nn.GELU, mlp_drop_rate: float = 0.,
                 drop_path_rate: _ratio_2_t = 0., layer_scale: float = 1e-5, add_identity: bool = True,
                 with_cp: bool = False):
        super().__init__()
        drop_path_rate = to_2tuple(drop_path_rate)

        self.cluster = Cluster(dim, dim, num_heads, head_dim, clusters_hw, proposals_hw, fold_factor, local_compute)
        self.mlp = MLP(dim, int(dim * mlp_ratio), out_dim, mlp_act_layer, mlp_drop_rate)
        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.drop_path1 = DropPath(drop_path_rate[0]) if drop_path_rate[0] > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate[1]) if drop_path_rate[1] > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_dim))
        self.add_identity = add_identity
        self.with_cp = with_cp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _forward(_x):
            if self.layer_scale:
                if self.add_identity:
                    _x = _x + self.drop_path1(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.cluster(self.norm1(_x)))
                    _x = _x + self.drop_path2(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(_x)))
                else:
                    _x = self.drop_path1(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.cluster(self.norm1(_x)))
                    _x = self.drop_path2(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(_x)))
            else:
                if self.add_identity:
                    _x = _x + self.drop_path1(self.cluster(self.norm1(_x)))
                    _x = _x + self.drop_path2(self.mlp(self.norm2(_x)))
                else:
                    _x = self.drop_path1(self.cluster(self.norm1(_x)))
                    _x = self.drop_path2(self.mlp(self.norm2(_x)))
            return _x

        if self.with_cp and x.requires_grad:
            x =  cp.checkpoint(_forward, x)
        else:
            x = _forward(x)
        return x


class ClusterStage(nn.Module):
    """Cluster Stage.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        num_blocks (int): Number of blocks in the stage. Default: 2.
        num_heads (int): Number of heads in context cluster. Default: 4.
        head_dim (int): Dimension of each head in context cluster. Default: 24.
        clusters_hw (tuple): Number of clusters (H W). Default: (2, 2).
        proposals_hw (tuple, optional): Number of cluster proposals (h w). Default: (2, 2).
        fold_factor (tuple): The fold factor of the feature map (feature map -> windows) (H W). Default: None.
        local_compute (bool): Whether to use local computation. Default: False.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        mlp_act_layer (Type[nn.Module]): Hidden activation layer in MLP. Default: nn.GELU.
        mlp_drop_rate (float | tuple): Dropout rate of MLP. Default: 0.
        drop_path_rate (tuple | list): Drop path rate. Default: (0.,).
        layer_scale (float): The scale of the layer. Default: 1e-5.
        add_identity (bool): Whether to add the identity tensor. Default: True.
        downsample (bool): Whether to downsample the feature maps. Default: True.
        with_cp (bool): Whether to use checkpoint. Default: False.
    """

    def __init__(self, dim: int, out_dim: int, num_blocks: int = 2, num_heads: int = 4,
                 head_dim: int = 24, clusters_hw: Tuple[int, int] = (2, 2),
                 proposals_hw: Optional[Tuple[int, int]] = (2, 2), fold_factor: Tuple[int, int] = (1, 1),
                 local_compute: bool = False, mlp_ratio=4., mlp_act_layer: Type[nn.Module] = nn.GELU,
                 mlp_drop_rate: float = 0., drop_path_rate: Union[Sequence[float], Sequence[Sequence[float]]] = (0.,),
                 layer_scale: float = 1e-5, add_identity: bool = True, downsample: bool = True, with_cp: bool = False):
        super().__init__()
        assert num_blocks >= 1
        self.downsample = downsample
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.blocks.append(ClusterBlock(dim, dim, num_heads, head_dim, clusters_hw, proposals_hw,
                                            fold_factor, local_compute, mlp_ratio, mlp_act_layer, mlp_drop_rate,
                                            drop_path_rate[i], layer_scale, add_identity, with_cp))

        if self.downsample:
            self.cluster_pool = ClusterPool(dim, out_dim, add_identity)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        for block in self.blocks:
            x = block(x)
        if self.downsample:
            x_down = self.cluster_pool(x)
            return x_down, x
        else:
            return x, x


class FECplus(nn.Module):
    """Feature Extractor with Clustering.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        num_classes (int): Number of classes for classification. Default: 1000.
        arch (str): Default architecture of the model. Default: 'S'.
        arch_setting (Optional[Sequence[list]]): Overwrite default architecture settings. Default: None.
        task (str): The task of the model. Default: 'classification'. Options: 'detection', 'segmentation'.
        out_indices (Sequence[int]): Output from which stages. Default: (3,).
        drop_path_rate (float | tuple): Drop path rate. Default: (0., 0.).
        with_positional_encoding (bool): Whether to add positional encoding. Default: True.
        act_layer (Type[nn.Module]): Activation layer. Default: nn.GELU.
        with_cp (bool): Whether to use checkpoint. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely, freeze running stats (mean and var).
            Note: Effect on Batch Norm and its variants only. Default: False.
        init_cfg (dict): Initialization config for mmdet/mmseg. Default: None.
        pretrained (str): Pretrained model or model path for mmdet/mmseg. Default: None.
    """

    classification_arch_settings = {
        # from left to right: (indices)
        # 'arch': embed_dim(0), num_blocks(1), num_heads(2), head_dim(3), clusters_hw(4),
        # proposals_hw(5), fold_factor(6), local_compute(7), mlp_ratio(8), mlp_drop_rate(9),
        # layer_scale(10), add_identity(11), downsample(12)
        'S': [[40, 2, 2, 20, (5, 5), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [80, 2, 4, 20, (4, 4), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [200, 6, 10, 20, (3, 3), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [320, 2, 16, 20, (2, 2), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'B': [[64, 2, 2, 32, (5, 5), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 2, 4, 32, (4, 4), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 6, 10, 32, (3, 3), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 2, 16, 32, (2, 2), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'L': [[64, 4, 2, 32, (5, 5), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 4, 4, 32, (4, 4), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 14, 10, 32, (3, 3), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 4, 14, 32, (2, 2), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],
    }

    detection_arch_settings = {
        'S': [[40, 2, 2, 20, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [80, 2, 4, 20, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [200, 6, 10, 20, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [320, 2, 16, 20, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'B': [[64, 2, 2, 32, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 2, 4, 32, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 6, 10, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 2, 16, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'L': [[64, 4, 2, 32, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 4, 4, 32, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 14, 10, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 4, 14, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],
    }

    segmentation_arch_settings = {
        'S': [[40, 2, 2, 20, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [80, 2, 4, 20, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [200, 6, 10, 20, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [320, 2, 16, 20, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'B': [[64, 2, 2, 32, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 2, 4, 32, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 6, 10, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 2, 16, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],

        'L': [[64, 4, 2, 32, (12, 12), (1, 1), (8, 8), True, 8., (0., 0.), 1e-5, True, True],
              [128, 4, 4, 32, (12, 12), (1, 1), (4, 4), True, 8., (0., 0.), 1e-5, True, True],
              [320, 14, 10, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, True],
              [512, 4, 14, 32, (12, 12), None, (1, 1), False, 4., (0., 0.), 1e-5, True, False]],
    }

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1000,
            arch: str = 'S',
            arch_setting: Optional[Sequence[list]] = None,
            task: str = 'classification',
            out_indices=(3,),
            drop_path_rate: _ratio_2_t = (0., 0.),
            with_positional_encoding: bool = True,
            act_layer: Type[nn.Module] = nn.GELU,
            with_cp: bool = False,
            frozen_stages: int = -1,
            norm_eval: bool = False,
            init_cfg=None,
            pretrained=None,
            **kwargs,
    ):
        super().__init__()

        self.task = task
        self.norm_eval = norm_eval
        self.with_positional_encoding = with_positional_encoding

        if self.task == 'classification':
            assert len(out_indices) == 1, 'For task classification, out_indices must be a single value.'
            arch_setting = arch_setting or self.classification_arch_settings[arch]
        elif self.task == 'detection':
            arch_setting = arch_setting or self.detection_arch_settings[arch]
        elif self.task == 'segmentation':
            arch_setting = arch_setting or self.segmentation_arch_settings[arch]
        else:
            raise NotImplementedError(f'Task {self.task} is not supported.')
        assert set(out_indices).issubset(i for i in range(len(arch_setting)))
        self.out_indices = out_indices

        if self.with_positional_encoding:
            self.in_channels = in_channels + 2
        else:
            self.in_channels = in_channels

        if frozen_stages not in range(-1, len(arch_setting)):
            raise ValueError(f'frozen_stages must be in range(-1, len(arch_setting)). But received {frozen_stages}')
        self.frozen_stages = frozen_stages

        self.aggregate = PointAggregation(self.in_channels, arch_setting[0][0], 4, 4, 0)

        depths = [x[1] for x in arch_setting]
        drop_path_rate = to_2tuple(drop_path_rate)
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate[0], sum(depths))]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate[1], sum(depths))]
        dpr = [(float(dpr1[i]), float(dpr2[i])) for i in range(sum(depths))]

        self.stages = nn.ModuleList()
        num_stages = len(arch_setting)
        for i, (embed_dim, num_blocks, num_heads, head_dim, clusters_hw, proposals_hw, fold_factor, local_compute,
                mlp_ratio, mlp_drop_rate, layer_scale, add_identity, downsample) in enumerate(arch_setting):
            if i == num_stages - 1:
                out_dim = embed_dim
            else:
                out_dim = arch_setting[i + 1][0]
            stage = ClusterStage(embed_dim, out_dim, num_blocks, num_heads, head_dim, clusters_hw, proposals_hw,
                                 fold_factor, local_compute, mlp_ratio, act_layer, mlp_drop_rate,
                                 dpr[sum(depths[:i]):sum(depths[:i+1])], layer_scale, add_identity, downsample, with_cp)
            self.stages.append(stage)

            if self.task != 'classification' and i in self.out_indices:
                norm = LayerNorm(embed_dim, data_format="channels_first")
                setattr(self, f'post_norm{i}', norm)

        if self.task == 'classification':
            self.num_classes = num_classes

            last_dim = arch_setting[self.out_indices[-1]][0]
            self.classifier_pre_norm = LayerNorm(last_dim, data_format="channels_last")
            self.head = nn.Linear(last_dim, self.num_classes)
            self.apply(self.pretrain_init_weights)
        else:
            self.init_cfg = copy.deepcopy(init_cfg)
            if self.init_cfg is not None or pretrained is not None:
                self.init_weights()

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        return x

    def forward_embeddings(self, x):
        b, _, h, w = x.shape
        if self.with_positional_encoding:
            y_coord = torch.arange(-0.5, h - 0.5) / (h - 1.0)
            x_coord = torch.arange(-0.5, w - 0.5) / (w - 1.0)
            grid = torch.stack(torch.meshgrid(y_coord, x_coord, indexing='ij'), dim=-1)
            x = torch.cat([x, grid.to(x).permute(2, 0, 1).unsqueeze(0).expand(b, -1, h, w)], dim=1)
        x = self.aggregate(x)
        return x

    def forward_tokens(self, x):
        if self.task == 'classification':
            for stage in self.stages:
                x, _ = stage(x)
            x = self.classifier_pre_norm(x.mean([-2, -1]))
            x = self.head(x)
            return x
        else:
            outs = []
            for i, stage in enumerate(self.stages):
                x, out = stage(x)
                if i in self.out_indices:
                    norm = getattr(self, f'post_norm{i}')
                    outs.append(norm(out))
            return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    @staticmethod
    def pretrain_init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0., std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if self.task == 'detection':
            logger = det_get_root_logger()
        elif self.task == 'segmentation':
            logger = seg_get_root_logger()
        else:
            from mmcv.utils import get_logger
            logger = get_logger(name='logger')

        if self.init_cfg is None and pretrained is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch.')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            assert 'checkpoint' in self.init_cfg, (f'Only support specify `Pretrained` in init_cfg` '
                                                   f'in {self.__class__.__name__}')

            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained
            else:
                raise ValueError('Either `init_cfg` or `pretrained` must be specified.')

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            init_state_dict = self.state_dict()
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k in init_state_dict:
                    if v.shape == init_state_dict[k].shape:
                        state_dict[k] = v
                    else:
                        if any(name in k for name in ['sim_alpha1', 'sim_alpha2', 'sim_beta1', 'sim_beta2', 'sim_alpha', 'sim_beta']):
                            src_tensor = v
                            tgt_tensor = init_state_dict[k].clone()
                            src_flatten = src_tensor.flatten()
                            tgt_flatten = tgt_tensor.flatten()
                            src_mean = src_flatten.mean()
                            tgt_len = tgt_flatten.numel()
                            src_len = src_flatten.numel()

                            if src_len >= tgt_len:
                                tgt_flatten.copy_(src_flatten[:tgt_len])
                            else:
                                tgt_flatten[:src_len].copy_(src_flatten)
                                tgt_flatten[src_len:].fill_(src_mean)
                            state_dict[k] = tgt_flatten.reshape_as(tgt_tensor)

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            logger.warning(f'missing_keys: {missing_keys}')
            logger.warning(f'unexpected_keys: {unexpected_keys}')

            for i in self.out_indices:
                if hasattr(self, f'post_norm{i}'):
                    constant_init(getattr(self, f'post_norm{i}'), val=1.0, bias=0.)
                    logger.info(f'Initialize the missing key self.post_norm{i} with constant 1.0 and bias 0.')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


# ----------------------------------------------imagenet1k_classification-----------------------------------------------
@register_model
def fec_plus_small(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = FECplus(arch='S', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


@register_model
def fec_plus_base(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = FECplus(arch='B', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


@register_model
def fec_plus_large(update_cfg: Dict[str, Any] = dict(crop_pct=0.95), **kwargs):
    model = FECplus(arch='L', **kwargs)  # drop_path_rate=(0.15, 0.15)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


# -----------------------------------------------cifar100_classification------------------------------------------------
@register_model
def fec_plus_small_cifar(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = FECplus(arch='S', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = CIFAR100_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


@register_model
def fec_plus_base_cifar(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = FECplus(arch='B', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = CIFAR100_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


@register_model
def fec_plus_large_cifar(update_cfg: Dict[str, Any] = dict(crop_pct=0.95), **kwargs):
    model = FECplus(arch='L', **kwargs)  # drop_path_rate=(0.15, 0.15)
    cfg = CIFAR100_CFG
    cfg.update(update_cfg)
    model.default_cfg = cfg
    return model


# ------------------------------------------------------detection-------------------------------------------------------
if has_mmdet:
    @det_BACKBONES.register_module()
    class fec_plus_det(FECplus):
        def __init__(self, *args, init_cfg=None, **kwargs):
            super().__init__(*args, init_cfg=init_cfg, **kwargs)


# -----------------------------------------------------segmentation-----------------------------------------------------
if has_mmseg:
    @seg_BACKBONES.register_module()
    class fec_plus_seg(FECplus):
        def __init__(self, *args, init_cfg=None, **kwargs):
            super().__init__(*args, init_cfg=init_cfg, **kwargs)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils import compute_throughput, get_flops

    imgs = torch.rand(4, 3, 224, 224).cuda()
    fec = fec_plus_small().cuda()
    print(fec)
    print(fec(imgs).shape)

    n_parameters = sum(p.numel() for p in fec.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters / 1024 ** 2))

    get_flops(fec)

    throughput = 0
    avg_throughput = 0
    for i in range(3):
        throughput = compute_throughput(fec, 256, 224)
        avg_throughput += throughput
        print(throughput)
    print("avg throughput: ", avg_throughput / 3)



    if has_mmdet:
        imgs = torch.rand(4, 3, 1280, 800).cuda()
        fec_det = fec_plus_det(
            arch='S',
            task='detection',
            out_indices=(0, 1, 2, 3),
            drop_path_rate=(0., 0.),
            with_positional_encoding=True,
            witch_cp=False,
        ).cuda()
        print(fec_det)
        outs = fec_det(imgs)
        for out in outs:
            print(out.shape)

    if has_mmseg:
        imgs = torch.rand(4, 3, 512, 512).cuda()
        fec_seg = fec_plus_seg(
            arch='S',
            task='segmentation',
            out_indices=(0, 1, 2, 3),
            drop_path_rate=(0., 0.),
            with_positional_encoding=True,
            witch_cp=False,
        ).cuda()
        print(fec_seg)
        outs = fec_seg(imgs)
        for out in outs:
            print(out.shape)
