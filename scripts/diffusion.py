import os
import math
import time
import warnings
import argparse
import numpy as np

import re

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp

from dataset.map_pre_old import MapDataset
from models.networks_2 import Denoiser
from utils.utils import plot_trajectories, sample_noise, embed_features
from utils.infer_2 import calculate_validation_loss_and_plot


def _bool(s):
    return str(s).lower() in {"1", "true", "yes", "y"}


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

    # Extract run number from log_dir (e.g., "runs_5" -> "5")

    eval_base_dir = os.path.join("results", args.model_type)
    os.makedirs(eval_base_dir, exist_ok=True)

    # Pattern to match run_*
    run_pattern = re.compile(r"run_(\d+)")
    run_numbers = []

    # Collect all run numbers
    for name in os.listdir(eval_base_dir):
        match = run_pattern.match(name)
        if match:
            run_numbers.append(int(match.group(1)))

    # Determine next run number
    next_run_num = max(run_numbers) + 1 if run_numbers else 0
    # Create new run folder
    new_run_folder = os.path.join(eval_base_dir, f"runs_{next_run_num}")
    os.makedirs(new_run_folder)



    
   
    writer = SummaryWriter(log_dir=new_run_folder)

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

    optimizer = optim.Adam(model.parameters(), lr=0.0)

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

                result = noised_tensor.clone()
                c_in_broadcast = c_in[:, None, None, None]
                result[:, :, args.obs_len:, :] = c_in_broadcast * noised_tensor[:, :, args.obs_len:, :]

                embedded = embed_features(result, sigma)
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
            ckpt_dir = os.path.join("checkpoints", f"{args.model_type}", f"runs_{next_run_num}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

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
            )
            print(f"Val @ epoch {epoch}: {avg_val_loss:.4f}")
            if not np.isnan(avg_val_loss):
                val_losses.append(avg_val_loss)
                writer.add_scalar("val/loss", avg_val_loss, epoch)

            if (epoch % (args.val_every)) == 0:
                if val_fig is not None:
          
                    results_dir = os.path.join(new_run_folder, "eval")
                    os.makedirs(results_dir, exist_ok=True)
                    try:
                        val_fig.savefig(os.path.join(results_dir, f"val_epoch_{epoch:03d}.png"), dpi=150, bbox_inches="tight")
                    except Exception:
                        warnings.warn("Could not save validation figure.")

    writer.close()
    print("Training completed.")


def build_argparser():
    p = argparse.ArgumentParser(description="Train diffusion model: 10 obs -> 20 pred")
    p.add_argument('--train_xml', type=str, default='./data/train', help='Training XML directory')
    p.add_argument('--val_xml', type=str, default='./data/val', help='Validation XML directory')
   
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

    p.add_argument('--val_every', type=int, default=20)

    p.add_argument('--model_type', type=str, default='diffusion', help='Model type for directory structure')
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
        if args.ckpt_path and os.path.isfile(args.ckpt_path):
            print(f"Loading checkpoint: {args.ckpt_path}")
            state = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(state)
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

#python train_diffusion.py --eval_only --val_xml ./data --batch_size 16 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --sigma_data 0.5 --ckpt_path ./checkpoints/model_epoch_0.pt --eval_save_dir ./eval_outputs

#python train_diffusion.py --train_xml ./data --val_xml ./data --epochs 50 --batch_size 16 --num_workers 4 --obs_len 10 --pred_len 20 --num_polylines 500 --num_points 10 --max_agents 32 --lr 3e-4 --val_every 20