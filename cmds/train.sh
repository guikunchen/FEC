CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --model fec_small -b 256 --lr 1e-3 --drop-path 0.1 --data_dir </path/to/imagenet> --amp  # using 8GPUs (128 images per GPU) or set lr to be 2e-3 might lead to better performance