# --------------------------------------------------------
# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --num_clu {number of clusters}
# --------------------------------------------------------

import models
import timm
import os
import torch
import argparse
import cv2
import time
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from einops import rearrange
import random
from timm.models import load_checkpoint
from torchvision.utils import draw_segmentation_masks
from torch_scatter import scatter_sum
from einops import rearrange
from sklearn.cluster import KMeans


object_categories = []
with open("./imagenet1k_id_to_label.txt", "r") as f:
    for line in f:
        _, val = line.strip().split(":")
        object_categories.append(val)

parser = argparse.ArgumentParser(description='FEC visualization')
parser.add_argument('--image', type=str, default="images/A.JPEG", help='path to image')
parser.add_argument('--shape', type=int, default=224, help='image size')
parser.add_argument('--model', default='coc_tiny_plain', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
parser.add_argument('--checkpoint', type=str, default="coc_tiny_plain.pth.tar", metavar='PATH', help='path to pretrained checkpoint')
parser.add_argument('--alpha', default=1., type=float, help='Transparent, 0-1')
# Note that FEC only results in 49 clusters in the final stage so that we have to adopt KMeans for easier inspection (see the second paragraph in Sec. 5.2).
# FEC may be sensitive when num_clu is a relatively small value, due to the KMeans algorithms.
parser.add_argument('--num_clu', type=int, default=3, help='number of clusters')
args = parser.parse_args()
assert args.model in timm.list_models(), "Please use a timm pre-trined model, see timm.list_models()"


# Preprocessing
def _preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,M,D]
    :param x2: [B,N,D]
    :return: similarity matrix [B,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.permute(0, 2, 1))
    return sim


def fwd_hook_fec(self, input, output):
    x = input[0]  # input tensor in a tuple
    value = self.conv_v(x)
    x = self.conv_f(x)
    assert self.fold_w == 1 and self.fold_h == 1

    b, c, w, h = x.shape
    centers = F.adaptive_avg_pool2d(x, (w // self.stride, h // self.stride))
    value_centers = rearrange(F.adaptive_avg_pool2d(value, (w // self.stride, h // self.stride)), 'b c w h -> b (w h) c')
    b, c, ww, hh = centers.shape
    sim = pairwise_cos_sim( centers.reshape(b, c, -1).permute(0, 2, 1), x.reshape(b, c, -1).permute(0, 2, 1) )  # [B,M,N]
    # we use mask to sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    global mask_layers
    if sim_max_idx.shape[0] == 1:
        mask_layers.append([sim_max_idx[0, 0, :].numpy().tolist(), rearrange(centers, 'b c w h -> b (w h) c')[0].numpy()])
    else:
        mask_layers.append([sim_max_idx[0, 0, :].numpy().tolist(), None])


def aggregate_masks(mask_layers, num_k=None):
    def _check(mask, k=224*224):
        total_lst = []
        for v in mask.values():
            total_lst.extend(v)
        total_set = set(total_lst)
        assert len(total_set) == k

    mask12 = {}
    idx = 0
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            # the clustering is based on 4x4 pixel patch (after standard conv)
            mask12[idx] = [i * 224 + j, i * 224 + j + 1, i * 224 + j + 2, i * 224 + j + 3, 
                (i + 1) * 224 + j,  (i + 1) * 224 + j + 1, (i + 1) * 224 + j + 2, (i + 1) * 224 + j + 3,
                (i + 2) * 224 + j,  (i + 2) * 224 + j + 1, (i + 2) * 224 + j + 2, (i + 2) * 224 + j + 3,
                (i + 3) * 224 + j,  (i + 3) * 224 + j + 1, (i + 3) * 224 + j + 2, (i + 3) * 224 + j + 3,
            ]
            idx += 1
    
    mask23 = {i: [] for i in range(56*56)}
    for i, j in enumerate(mask_layers[0][0]):
        mask23[j].extend(mask12[i])
    
    mask34 = {i: [] for i in range(28*28)}
    for i, j in enumerate(mask_layers[1][0]):
        mask34[j].extend(mask23[i])

    mask45 = {i: [] for i in range(7*7)}
    for i, j in enumerate(mask_layers[2][0]):
        mask45[j].extend(mask34[i])

    # _check(mask45)
    non_empty_clusters, idx_map = [], {}
    for i in range(7*7):
        if len(mask45[i]) > 0:
            non_empty_clusters.append(i)
            idx_map[len(idx_map)] = i
    
    if num_k is None or num_k >= len(non_empty_clusters):
        final_mask = torch.zeros((1, len(non_empty_clusters), 224 * 224))
        for idx1, idx2 in enumerate(non_empty_clusters):
            final_mask[0, idx1, mask45[idx2]] = 1
        final_mask = final_mask.reshape(1, len(non_empty_clusters), 224, 224)
    else:
        mask56 = {i: [] for i in range(num_k)}
        feats = mask_layers[2][1][non_empty_clusters]
        kmeans = KMeans(n_clusters=num_k, random_state=0, n_init="auto").fit(feats)
        for idx1, idx2 in enumerate(kmeans.labels_):
            mask56[idx2].extend(mask45[idx_map[idx1]])
        # _check(mask56)
        final_mask = torch.zeros((1, num_k, 224 * 224))
        for idx1, idx2 in enumerate(range(num_k)):
            final_mask[0, idx1, mask56[idx2]] = 1
        final_mask = final_mask.reshape(1, num_k, 224, 224)

    return final_mask


@torch.no_grad()
def infer(model, img_path, num_k=None):
    global mask_layers
    mask_layers = []
    image, raw_image = _preprocess(img_path)
    image = image.unsqueeze(dim=0)
    out = model(image)
    if type(out) is tuple: out = out[0]
    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'==> Prediction is: {object_categories[index]} possibility: {possibility * 100:.3f}%')
    os.makedirs(f"images/cluster_vis/{args.model}", exist_ok=True)
    image_name = os.path.basename(img_path).split(".")[0]
    from torchvision.io import read_image
    img = read_image(img_path)

    mask = aggregate_masks(mask_layers, num_k)
    mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5

    # randomly selected some good colors.
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue
    if mask.shape[0] <= len(colors):
        colors = colors[:mask.shape[0]]
    else:
        colors = (colors * (mask.shape[0] // 16 + 1))[:mask.shape[0]]
        random.seed(123)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = f"images/cluster_vis/{args.model}/{image_name}_{time.time()}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")


def main():
    model = timm.create_model(model_name=args.model, pretrained=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, True)
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError
    model.network[1].register_forward_hook(fwd_hook_fec)
    model.network[3].register_forward_hook(fwd_hook_fec)
    model.network[5].register_forward_hook(fwd_hook_fec)

    infer(model, args.image, args.num_clu)


if __name__ == '__main__':
    main()
