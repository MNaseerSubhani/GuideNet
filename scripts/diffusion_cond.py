import os
import math
import time
import warnings
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp

from map_pre_old import MapDataset
from networks_2 import Denoiser
from utils import embed_features, sample_noise
from infer_2 import calculate_validation_loss_and_plot, direction_onehot_from_theta


EMBED_DX = 1280  # must match networks_2.FeatureMLP(input_dim)


class CondProjAdditive(torch.nn.Module):
    """Maps direction one-hot [B,A,3] -> embedding [B,A,EMBED_DX] for additive conditioning."""
    def __init__(self, out_dim: int = EMBED_DX):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, out_dim)
        )

    def forward(self, y_onehot: torch.Tensor) -> torch.Tensor:
        # y_onehot: [B,A,3] or [B,3]
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)
        return self.net(y_onehot)


def create_dataloaders(xml_dir: str, batch_size: int, num_workers: int,
                       obs_len: int, pred_len: int,
                       max_radius: int, num_timesteps: int,
                       num_polylines: int, num_points: int,
                       max_agents: int):
    dataset = MapDataset(
        xml_dir=xml_dir,
        obs_len=obs_len, pred_len=pred_len, max_radius=max_radius,
        num_timesteps=num_timesteps, num_polylines=num_polylines, num_points=num_points,
        save_plots=False, max_agents=max_agents
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    return dataset, dataloader


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    dataset, dataloader = create_dataloaders(
        xml_dir=args.train_xml,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_radius=args.max_radius,
        num_timesteps=args.obs_len + args.pred_len,
        num_polylines=args.num_polylines,
        num_points=args.num_points,
        max_agents=args.max_agents,
    )

    model = Denoiser().to(device)
    model.train()

    # conditioning projector (additive over future embedding)
    cond_proj = CondProjAdditive(EMBED_DX).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.add_param_group({"params": cond_proj.parameters(), "lr": 0.0})  # same LR schedule

    steps_per_epoch = max(1, len(dataloader))
    ramp_up_steps = max(1, int(args.lr_warmup_frac * steps_per_epoch))
    target_lr = args.lr
    sigma_data = args.sigma_data

    print(f"Dataset size: {len(dataset)} | Steps/epoch: {steps_per_epoch}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    global_step = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            (ego_ids, feature_tensor, feature_mask,
             roadgraph_tensor, roadgraph_mask,
             observed, observed_masks,
             ground_truth, ground_truth_masks,
             scene_means, scene_stds) = batch

            feature_tensor = feature_tensor.to(device, non_blocking=True)
            feature_mask   = feature_mask.to(device, non_blocking=True)
            roadgraph_tensor = roadgraph_tensor.to(device, non_blocking=True)
            roadgraph_mask   = roadgraph_mask.to(device, non_blocking=True)
            scene_means = scene_means.to(device, non_blocking=True)
            scene_stds  = scene_stds.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                sigma, noised_tensor = sample_noise(feature_tensor)

                c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
                c_out  = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
                c_in   = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

                # precondition only future time steps
                result = noised_tensor.clone()
                c_in_broadcast = c_in[:, None, None, None]
                result[:, :, args.obs_len:, :] = c_in_broadcast * noised_tensor[:, :, args.obs_len:, :]

                # base embedding
                embedded = embed_features(result, sigma)

                # direction one-hot from future theta (teacher forcing during training)
                theta_future = feature_tensor[:, :, args.obs_len:, 2]
                y_onehot = direction_onehot_from_theta(theta_future, turn_thresh_deg=args.turn_thresh_deg)
                c_embed = cond_proj(y_onehot)  # [B,A,EMBED_DX]
                # expand to future horizon and add only on future steps
                B, A, Tp = theta_future.shape
                cond_time = c_embed.unsqueeze(2).expand(B, A, Tp, EMBED_DX)
                embedded[:, :, args.obs_len:, :] = embedded[:, :, args.obs_len:, :] + cond_time

                model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, args.obs_len:, :]

                gt_pred   = feature_tensor[:, :, args.obs_len:, :]
                mask_pred = feature_mask[:, :, args.obs_len:]
                valid_mask = mask_pred.unsqueeze(-1).expand_as(gt_pred)

                recon = model_out * c_out[:, None, None, None] + noised_tensor[:, :, args.obs_len:, :] * c_skip[:, None, None, None]

                squared_diff = (recon - gt_pred) ** 2
                masked_squared_diff = squared_diff * valid_mask.float()
                loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)

                weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
                loss = (loss_per_batch * weight).mean()

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            lr = (global_step / ramp_up_steps) * target_lr if global_step < ramp_up_steps else target_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            writer.add_scalar("train/iter_loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(lr), global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            writer.add_scalar("train/sigma_example0", float(sigma[0].item()), global_step)

            train_losses.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} - loss={avg_epoch_loss:.4f} - {time.time()-t0:.1f}s")

        if (epoch % args.val_every) == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"model_epoch_{epoch}.pt")
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt = {
                "model": model.state_dict(),
                "cond_proj": cond_proj.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, ckpt_path)

            avg_val_loss, val_fig = calculate_validation_loss_and_plot(
                model=model,
                val_xml_dir=args.val_xml,
                val_batch_size=args.batch_size,
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                max_radius=args.max_radius,
                num_polylines=args.num_polylines,
                num_points=args.num_points,
                max_agents=args.max_agents,
                sigma_data=args.sigma_data,
                device=device,
                direction_command=True,
                cond_scale=args.cond_scale,
                turn_thresh_deg=args.turn_thresh_deg,
                cond_proj_state_dict=cond_proj.state_dict(),
            )
            print(f"Val @ epoch {epoch}: {avg_val_loss:.4f}")
            if not np.isnan(avg_val_loss):
                val_losses.append(avg_val_loss)
                writer.add_scalar("val/loss", avg_val_loss, epoch)
            if (epoch % (args.val_plot_every)) == 0:
                if val_fig is not None:
                    ts_dir = os.path.join(args.log_dir, "val_plots")
                    os.makedirs(ts_dir, exist_ok=True)
                    try:
                        val_fig.savefig(os.path.join(ts_dir, f"val_epoch_{epoch:03d}.png"), dpi=150, bbox_inches="tight")
                    except Exception:
                        warnings.warn("Could not save validation figure.")

    writer.close()
    print("Training completed.")


def build_argparser():
    p = argparse.ArgumentParser(description="Train diffusion model with maneuver conditioning (L/R/S)")
    p.add_argument('--train_xml', type=str, default='./data', help='Training XML directory')
    p.add_argument('--val_xml', type=str, default='./data', help='Validation XML directory')
    p.add_argument('--log_dir', type=str, default='./runs_5/REAL', help='TensorBoard log dir')
    p.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Checkpoint directory')

    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--obs_len', type=int, default=10)
    p.add_argument('--pred_len', type=int, default=20)
    p.add_argument('--max_radius', type=int, default=100)
    p.add_argument('--num_polylines', type=int, default=500)
    p.add_argument('--num_points', type=int, default=10)
    p.add_argument('--max_agents', type=int, default=32)

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr_warmup_frac', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--sigma_data', type=float, default=0.5)
    p.add_argument('--turn_thresh_deg', type=float, default=15.0, help='Angle threshold to classify L/R/S')

    p.add_argument('--val_every', type=int, default=20)
    p.add_argument('--val_plot_every', type=int, default=100)

    # Conditioning guidance for eval
    p.add_argument('--cond_scale', type=float, default=2.0, help='Classifier-free guidance scale during eval')

    # Evaluation options
    p.add_argument('--eval_only', action='store_true', help='Run evaluation only (no training)')
    p.add_argument('--ckpt_path', type=str, default='', help='Path to model checkpoint for eval')
    p.add_argument('--eval_save_dir', type=str, default='./eval_outputs', help='Where to save eval plots')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Denoiser().to(device)
        cond_proj = CondProjAdditive(EMBED_DX).to(device)
        
        if args.ckpt_path and os.path.isfile(args.ckpt_path):
            print(f"Loading checkpoint: {args.ckpt_path}")
            state = torch.load(args.ckpt_path, map_location=device)
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"])
                if "cond_proj" in state:
                    cond_proj.load_state_dict(state["cond_proj"])
                    print("Loaded cond_proj from checkpoint")
                else:
                    print("Warning: cond_proj not found in checkpoint")
            else:
                # Legacy checkpoint format (only model state dict)
                model.load_state_dict(state)
                print("Warning: Legacy checkpoint format - cond_proj not loaded")
        else:
            if args.ckpt_path:
                print(f"Warning: checkpoint not found at {args.ckpt_path}. Evaluating with randomly initialized model.")

        os.makedirs(args.eval_save_dir, exist_ok=True)
        avg_val_loss, val_fig = calculate_validation_loss_and_plot(
            model=model,
            val_xml_dir=args.val_xml,
            val_batch_size=args.batch_size,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            max_radius=args.max_radius,
            num_polylines=args.num_polylines,
            num_points=args.num_points,
            max_agents=args.max_agents,
            sigma_data=args.sigma_data,
            device=device,
            direction_command=True,
            cond_scale=args.cond_scale,
            turn_thresh_deg=args.turn_thresh_deg,
            cond_proj_state_dict=cond_proj.state_dict(),
        )
        print(f"Eval avg loss: {avg_val_loss:.6f}")
        if val_fig is not None:
            try:
                out_path = os.path.join(args.eval_save_dir, "eval_plot.png")
                val_fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Saved eval plot -> {out_path}")
            except Exception:
                warnings.warn("Could not save eval plot.")
    else:
        train_loop(args)

# Example usage:
# Training:
# python train_diffusion_cond.py --train_xml ./data --val_xml ./data --epochs 50 --batch_size 16 --num_workers 4 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --lr 3e-4 --val_every 20
# Eval:
# python train_diffusion_cond.py --eval_only --val_xml ./data --batch_size 16 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --sigma_data 0.5 --ckpt_path ./checkpoints/model_epoch_0.pt --eval_save_dir ./eval_outputs


