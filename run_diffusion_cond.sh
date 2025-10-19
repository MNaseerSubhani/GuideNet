#!/bin/bash

# GuideNet Conditional Diffusion runner
# Usage: ./run_diffusion_cond.sh train
#        ./run_diffusion_cond.sh eval

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [ "$1" = "train" ]; then
    python scripts/diffusion_cond.py --train_xml ./data/train/ --val_xml ./data/val/ --epochs 1000 --batch_size 16 --num_workers 4 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --lr 3e-4 --val_every 100  --turn_thresh_deg 15.0 --cond_scale 2.0 --model_type diffusion_cond
elif [ "$1" = "eval" ]; then
    python scripts/diffusion_cond.py --eval_only --val_xml ./data/val/ --batch_size 16 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --sigma_data 0.5 --turn_thresh_deg 15.0 --cond_scale 2.0
else
    echo "Usage: $0 [train|eval]"
    exit 1
fi
#--model_type diffusion_cond