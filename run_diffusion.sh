#!/bin/bash

# Simple GuideNet runner
# Usage: ./run_diffusion.sh train
#        ./run_diffusion.sh eval

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [ "$1" = "train" ]; then
    python scripts/diffusion.py --train_xml ./data/train/ --val_xml ./data/val/ --epochs 1000 --batch_size 16 --num_workers 4 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --lr 3e-4 --val_every 100 --model_type diffusion
elif [ "$1" = "eval" ]; then
    python scripts/diffusion.py --eval_only --val_xml ./data/val/ --batch_size 16 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --sigma_data 0.5 
else
    echo "Usage: $0 [train|eval]"
    exit 1
fi