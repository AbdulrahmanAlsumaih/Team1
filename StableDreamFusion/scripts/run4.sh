#! /bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial_rabbit2 --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial2_rabbit2 --dmtet --iters 5000 --init_with trial_rabbit2/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a corgi dog, highly detailed 3d model" --workspace trial_corgi --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a corgi dog, highly detailed 3d model" --workspace trial2_corgi --dmtet --iters 5000 --init_with trial_corgi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial_cactus --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial2_cactus --dmtet --iters 5000 --init_with trial_cactus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "the leaning tower of Pisa" --workspace trial_pisa --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "the leaning tower of Pisa" --workspace trial2_pisa --dmtet --iters 5000 --init_with trial_pisa/checkpoints/df.pth