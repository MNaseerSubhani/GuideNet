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
│   ├── network_diffusion.py        # Unconditional diffusion model
│   └── networks_cond_diffusion.py  # Conditional diffusion model
├── scripts/                # Training and evaluation scripts
│   ├── diffusion.py        # Unconditional diffusion training
│   ├── diffusion_cond.py   # Conditional diffusion training
│   └── infer_diffusions.py # Inference and evaluation utilities
├── utils/                  # Utility functions
│   └── utils.py           # General utilities
├── checkpoints/            # Model checkpoints
│   ├── diffusion/         # Unconditional model checkpoints
│   └── diffusion_cond/    # Conditional model checkpoints
├── results/               # Training logs and outputs
│   ├── diffusion/         # Unconditional model results
│   └── diffusion_cond/    # Conditional model results
├── run_diffusion.sh       # Unconditional diffusion runner
└── run_diffusion_cond.sh  # Conditional diffusion runner
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
pip install -r requirements.txt
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
# Train unconditional diffusion model
bash run_diffusion.sh train

# Evaluate trained model
bash run_diffusion.sh eval

# Train conditional diffusion model with [Right, left, Straight] conditions
bash run_diffusion_cond.sh train

# Evaluate trained model
bash run_diffusion_cond.sh eval
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



## Output Files

### Training Outputs
- **Checkpoints**: Saved in `checkpoints/{model_type}/runs_{N}/`
- **Logs**: TensorBoard logs in `results/{model_type}/run_{N}/train/`
- **Validation Plots**: Saved in `results/{model_type}/run_{N}/eval/`

