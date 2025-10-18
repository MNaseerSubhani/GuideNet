# GuideNet: Diffusion-based Trajectory Prediction

GuideNet is a diffusion-based trajectory prediction model for autonomous driving scenarios. The project implements both unconditional and conditional diffusion models for predicting vehicle trajectories with maneuver conditioning (Left/Right/Straight).

## Project Structure

```
GuideNet/
├── data/                    # Training and validation data
│   ├── train/              # Training XML files
│   └── val/                # Validation XML files
├── dataset/                # Data loading and preprocessing
│   └── map_pre_old.py      # MapDataset implementation
├── models/                 # Neural network architectures
│   └── networks_2.py       # Denoiser model implementation
├── scripts/                # Training and evaluation scripts
│   ├── diffusion.py        # Unconditional diffusion training
│   ├── diffusion_cond.py   # Conditional diffusion training
│   └── real_train.py      # Additional training utilities
├── utils/                  # Utility functions
│   ├── utils.py           # General utilities
│   └── infer_2.py         # Inference and evaluation utilities
├── checkpoints/            # Model checkpoints
├── results/               # Training logs and outputs
└── run_diffusion.sh       # Convenience script for running experiments
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GuideNet
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tensorboard
```

3. Set up the Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Data Setup

### Data Directory Structure

Place your XML trajectory data files in the following structure:

```
data/
├── train/                 # Training data
│   ├── SGP_NUP-0_1_T-1.xml
│   ├── SGP_NUP-1_1_T-1.xml
│   ├── SGP_NUP-3_1_T-1.xml
│   └── SGP_NUP-4_1_T-1.xml
└── val/                   # Validation data
    └── SGP_NUP-2_1_T-1.xml
```

### XML Data Format

The XML files should contain trajectory data with the following structure:
- Agent trajectories with timesteps, positions (x, y), and orientations (theta)
- Roadgraph information for map context
- Dynamic obstacle information

## Usage

### Quick Start

Use the convenience script for quick training and evaluation:

```bash
# Make the script executable
chmod +x run_diffusion.sh

# Train unconditional diffusion model
./run_diffusion.sh train

# Evaluate trained model
./run_diffusion.sh eval
```

### Training Process

#### Step 1: Unconditional Diffusion Training (`diffusion.py`)

Train the base diffusion model without conditioning:

```bash
python scripts/diffusion.py \
    --train_xml ./data/train/ \
    --val_xml ./data/val/ \
    --epochs 200 \
    --batch_size 32 \
    --num_workers 4 \
    --obs_len 10 \
    --pred_len 20 \
    --num_polylines 500 \
    --num_points 10 \
    --max_agents 32 \
    --lr 3e-4 \
    --val_every 20 \
    --model_type diffusion
```

**Key Parameters:**
- `--train_xml`: Path to training XML files
- `--val_xml`: Path to validation XML files
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--obs_len`: Observation length (timesteps)
- `--pred_len`: Prediction length (timesteps)
- `--num_polylines`: Maximum number of polylines in roadgraph
- `--num_points`: Points per polyline
- `--max_agents`: Maximum agents per scene
- `--lr`: Learning rate
- `--val_every`: Validation frequency (epochs)

#### Step 2: Conditional Diffusion Training (`diffusion_cond.py`)

Train the conditional diffusion model with maneuver conditioning:

```bash
python scripts/diffusion_cond.py \
    --train_xml ./data/train/ \
    --val_xml ./data/val/ \
    --log_dir ./runs_conditional/ \
    --ckpt_dir ./checkpoints/conditional/ \
    --epochs 200 \
    --batch_size 32 \
    --num_workers 4 \
    --obs_len 10 \
    --pred_len 20 \
    --num_polylines 500 \
    --num_points 10 \
    --max_agents 32 \
    --lr 3e-4 \
    --val_every 20 \
    --turn_thresh_deg 15.0 \
    --cond_scale 2.0
```

**Additional Parameters for Conditional Model:**
- `--turn_thresh_deg`: Angle threshold for Left/Right/Straight classification
- `--cond_scale`: Classifier-free guidance scale for evaluation
- `--log_dir`: Directory for TensorBoard logs
- `--ckpt_dir`: Directory for model checkpoints

### Evaluation

#### Evaluate Unconditional Model

```bash
python scripts/diffusion.py \
    --eval_only \
    --val_xml ./data/val/ \
    --batch_size 16 \
    --obs_len 10 \
    --pred_len 20 \
    --num_polylines 500 \
    --num_points 10 \
    --max_agents 32 \
    --sigma_data 0.5 \
    --ckpt_path ./checkpoints/diffusion/runs_2/model_epoch_100.pt \
    --eval_save_dir ./eval_outputs
```

#### Evaluate Conditional Model

```bash
python scripts/diffusion_cond.py \
    --eval_only \
    --val_xml ./data/val/ \
    --batch_size 16 \
    --obs_len 10 \
    --pred_len 20 \
    --num_polylines 500 \
    --num_points 10 \
    --max_agents 32 \
    --sigma_data 0.5 \
    --ckpt_path ./checkpoints/conditional/model_epoch_100.pt \
    --eval_save_dir ./eval_outputs \
    --cond_scale 2.0 \
    --turn_thresh_deg 15.0
```

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir results/diffusion/
```

## Model Architecture

### Unconditional Diffusion Model
- **Denoiser**: Transformer-based architecture with agent self-attention
- **Input**: Observed trajectories + roadgraph context
- **Output**: Predicted future trajectories
- **Loss**: Weighted MSE with sigma-dependent weighting

### Conditional Diffusion Model
- **Base Model**: Same denoiser as unconditional model
- **Conditioning**: Additive conditioning with maneuver classification (L/R/S)
- **Conditioning Projector**: MLP that maps direction one-hot to embedding
- **Training**: Teacher forcing with ground truth maneuvers
- **Inference**: Classifier-free guidance for controllable generation

## Key Features

1. **Diffusion-based Generation**: Uses EDM (Elucidated Diffusion Models) framework
2. **Multi-agent Support**: Handles multiple agents per scene
3. **Roadgraph Integration**: Incorporates map context for better predictions
4. **Maneuver Conditioning**: Conditional generation for Left/Right/Straight maneuvers
5. **Ego-centric Processing**: Transforms trajectories to ego vehicle perspective
6. **Automatic Mixed Precision**: Uses AMP for faster training on GPU

## Output Files

### Training Outputs
- **Checkpoints**: Saved in `checkpoints/{model_type}/runs_{N}/`
- **Logs**: TensorBoard logs in `results/{model_type}/run_{N}/`
- **Validation Plots**: Saved in `results/{model_type}/run_{N}/eval/`

### Evaluation Outputs
- **Plots**: Trajectory visualizations saved to specified `--eval_save_dir`
- **Metrics**: Validation loss printed to console

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `--batch_size` or `--max_agents`
2. **Data Loading Errors**: Ensure XML files are in correct format and paths are correct
3. **Import Errors**: Make sure `PYTHONPATH` includes the project root
4. **Checkpoint Loading**: Verify checkpoint paths and file existence

### Performance Tips

1. **GPU Memory**: Use smaller batch sizes for limited GPU memory
2. **Data Loading**: Increase `--num_workers` for faster data loading (if CPU allows)
3. **Mixed Precision**: Training automatically uses AMP when CUDA is available
4. **Validation Frequency**: Adjust `--val_every` based on training time

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{guidenet2024,
  title={GuideNet: Diffusion-based Trajectory Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
