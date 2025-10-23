







import os
import math
import time
import warnings
import argparse
import numpy as np

import re

import os
import glob
import re

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp

from dataset.map_pre_old import MapDataset
from models.networks_cond_diffusion import Denoiser
from utils.utils import embed_features, sample_noise
from scripts.infer_diffusions import calculate_validation_loss_and_plot, direction_onehot_from_theta
import torch.nn.functional as F


EMBED_DX = 1280  # must match networks_2.FeatureMLP(input_dim)


# class CondProjAdditive(torch.nn.Module):
#     """Maps direction one-hot [B,A,3] -> embedding [B,A,EMBED_DX] for additive conditioning."""
#     def __init__(self, out_dim: int = EMBED_DX):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(3, 128), torch.nn.SiLU(),
#             torch.nn.Linear(128, out_dim)
#         )

#     def forward(self, y_onehot: torch.Tensor) -> torch.Tensor:
#         # y_onehot: [B,A,3] or [B,3]
#         if y_onehot.dim() == 2:
#             y_onehot = y_onehot.unsqueeze(1)
#         return self.net(y_onehot)


# training script: replace CondProjAdditive with this:

class CondProjConcat(torch.nn.Module):
    """
    Maps direction one-hot [B,A,3] -> embedding [B,A,EMBED_DX] for concatenation conditioning.
    Also includes a learned null embedding used when we drop the condition (classifier-free).
    """
    def __init__(self, out_dim: int = EMBED_DX):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, out_dim)
        )
        # learned null token for unconditional examples (shape: [1, out_dim])
        self.null_token = torch.nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, y_onehot: torch.Tensor) -> torch.Tensor:
        # y_onehot: [B,A,3] or [B,3]
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)  # [B,1,3]
        emb = self.net(y_onehot)  # [B,A,out_dim]
        return emb  # caller handles replacing with null_token for dropped condition



def _bool(s):
    return str(s).lower() in {"1", "true", "yes", "y"}


def compute_road_boundary_distance(predictions, roadgraph_tensor, roadgraph_mask, scene_std):
    """
    Compute distance from predicted trajectories to nearest road boundaries.
    
    Args:
        predictions: [B, A, T, 3] - predicted trajectories (x, y, theta)
        roadgraph_tensor: [B, RG, NUM_POINTS, 2] - road polylines (x, y)
        roadgraph_mask: [B, RG] - mask for valid polylines
        scene_std: [3] - standard deviation for scaling (x, y, theta)
    
    Returns:
        distances: [B, A, T] - minimum distance to road boundaries
    """
    B, A, T, _ = predictions.shape
    B, RG, NUM_POINTS, _ = roadgraph_tensor.shape

    # Extract and rescale predicted (x, y)
    pred_xy = predictions[..., :2]  # [B, A, T, 2]
    scale_xy = scene_std[:2]
    pred_xy = pred_xy / scale_xy[None, None, None, :]  # back to world coords

    # Initialize output
    min_distances = torch.full((B, A, T), 1e3, device=predictions.device)

    for b in range(B):
        mask = roadgraph_mask[b]
        if not mask.any():
            continue

        road_points = roadgraph_tensor[b, mask]  # [N_valid, NUM_POINTS, 2]
        road_points = road_points.reshape(-1, 2)  # flatten to all boundary points

        preds = pred_xy[b].reshape(-1, 2)  # [A*T, 2]
        distances = torch.cdist(preds, road_points, p=2)  # [A*T, N_points]
        min_dists = distances.min(dim=1)[0]  # [A*T]

        min_distances[b] = min_dists.view(A, T)

    return min_distances


# def compute_road_boundary_distance(predictions, roadgraph_tensor, roadgraph_mask, scene_std):
#     """
#     Compute distance from predicted trajectories to road boundaries.
    
#     Args:
#         predictions: [B, A, T, 3] - predicted trajectories (x, y, theta)
#         roadgraph_tensor: [B, RG, NUM_POINTS, 2] - road polylines (x, y)
#         roadgraph_mask: [B, RG] - mask for valid polylines
#         scene_std: [3] - standard deviation for scaling (x, y, theta)
    
#     Returns:
#         distances: [B, A, T] - minimum distance to road boundaries
#     """
#     B, A, T, _ = predictions.shape
#     B, RG, NUM_POINTS, _ = roadgraph_tensor.shape
    
#     # Extract x, y coordinates from predictions
#     pred_xy = predictions[:, :, :, :2]  # [B, A, T, 2]
    
#     # Scale predictions back to original coordinates for distance calculation
#     scale_xy = scene_std[:2]  # [2]
#     pred_xy_original = pred_xy / scale_xy[None, None, None, :]
    
#     # Initialize distances with large values
#     min_distances = torch.full((B, A, T), 1000.0, device=predictions.device)
    
#     # Process each batch element separately to handle different roadgraph masks
#     for b in range(B):
#         # Get valid road polylines for this batch element
#         batch_roadgraph = roadgraph_tensor[b]  # [RG, NUM_POINTS, 2]
#         batch_mask = roadgraph_mask[b]  # [RG]
        
#         if not batch_mask.any():
#             continue  # No valid roads for this batch element
            
#         valid_roads = batch_roadgraph[batch_mask]  # [N_valid_roads, NUM_POINTS, 2]
        
#         # Scale road polylines back to original coordinates
#         valid_roads_original = valid_roads / scale_xy[None, None, :]
        
#         # Get predictions for this batch element
#         batch_pred_xy = pred_xy_original[b]  # [A, T, 2]
        
#         # Reshape for distance calculation
#         pred_flat = batch_pred_xy.view(A * T, 1, 2)  # [A*T, 1, 2]
#         roads_flat = valid_roads_original.view(1, -1, 2)  # [1, N_valid_roads*NUM_POINTS, 2]
        
#         # Compute pairwise distances
#         distances = torch.cdist(pred_flat, roads_flat, p=2)  # [A*T, 1, N_road_points]
        
#         # Find minimum distance for each prediction point
#         batch_min_distances = distances.min(dim=-1)[0]  # [A*T]
        
#         # Reshape and store
#         min_distances[b] = batch_min_distances.view(A, T)
    
#     return min_distances


def road_boundary_mask_loss(predictions, roadgraph_tensor, roadgraph_mask, scene_std, 
                           boundary_threshold=5.0, loss_weight=1.0):
    """
    Compute mask loss that penalizes predictions outside road boundaries.
    
    Args:
        predictions: [B, A, T, 3] - predicted trajectories
        roadgraph_tensor: [B, RG, NUM_POINTS, 2] - road polylines
        roadgraph_mask: [B, RG] - mask for valid polylines
        scene_std: [3] - standard deviation for scaling
        boundary_threshold: float - distance threshold for road boundaries
        loss_weight: float - weight for this loss component
    
    Returns:
        loss: scalar tensor - road boundary mask loss
    """
    distances = compute_road_boundary_distance(predictions, roadgraph_tensor, roadgraph_mask, scene_std)
    
    # Create mask for points outside road boundaries
    outside_mask = distances > boundary_threshold  # [B, A, T]
    
    # Compute loss only for points outside boundaries
    if outside_mask.any():
        # Penalty increases with distance from road
        penalty = torch.clamp(distances - boundary_threshold, min=0.0)  # [B, A, T]
        penalty = penalty * outside_mask.float()  # Only apply to outside points
        
        # Average loss over all outside points
        loss = penalty.sum() / (outside_mask.sum() + 1e-6)  # Avoid division by zero
        return loss_weight * loss
    else:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)



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


def find_latest_checkpoint(model_type="diffusion_cond"):
    """
    Find the latest checkpoint from checkpoints/{model_type}/{latest_runs}/{latest_model}
    Returns the path to the latest checkpoint file.
    """
    checkpoint_base = os.path.join("checkpoints", model_type)
    
    if not os.path.exists(checkpoint_base):
        return None
    
    # Find all runs directories
    runs_dirs = []
    for item in os.listdir(checkpoint_base):
        if os.path.isdir(os.path.join(checkpoint_base, item)) and item.startswith("runs_"):
            runs_dirs.append(item)
    
    if not runs_dirs:
        return None
    
    # Sort runs directories by number (runs_0, runs_1, etc.)
    def extract_run_number(run_dir):
        match = re.search(r'runs_(\d+)', run_dir)
        return int(match.group(1)) if match else 0
    
    latest_run_dir = max(runs_dirs, key=extract_run_number)
    latest_run_path = os.path.join(checkpoint_base, latest_run_dir)
    
    # Find all checkpoint files in the latest run
    checkpoint_files = glob.glob(os.path.join(latest_run_path, "model_epoch_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and get the latest
    def extract_epoch_number(checkpoint_file):
        match = re.search(r'model_epoch_(\d+)\.pt', os.path.basename(checkpoint_file))
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch_number)
    return latest_checkpoint


def create_eval_save_dir(model_type="diffusion_cond"):
    """
    Create evaluation save directory in format: results/{model_type}/eval_{int}/
    Returns the path to the created directory.
    """
    results_base = os.path.join("results", model_type, 'eval')
    os.makedirs(results_base, exist_ok=True)
    
    # Find existing eval directories
    eval_dirs = []
    for item in os.listdir(results_base):
        if os.path.isdir(os.path.join(results_base, item)) and item.startswith("eval_"):
            eval_dirs.append(item)
    
    # Determine next eval number
    if eval_dirs:
        def extract_eval_number(eval_dir):
            match = re.search(r'eval_(\d+)', eval_dir)
            return int(match.group(1)) if match else 0
        
        next_eval_num = max(extract_eval_number(eval_dir) for eval_dir in eval_dirs) + 1
    else:
        next_eval_num = 0
    
    eval_dir = os.path.join(results_base, f"eval_{next_eval_num}")
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir




def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    eval_base_dir = os.path.join("results", args.model_type, 'train')
    os.makedirs(eval_base_dir, exist_ok=True)

    # Pattern to match run_*
    run_pattern = re.compile(r"runs_(\d+)")
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

    model = Denoiser(embed_dim=EMBED_DX).to(device)

    # state = torch.load("/home/tic/Desktop/GuideNet/checkpoints/diffusion_cond/runs_18/model_epoch_700.pt", map_location=device)
    

    model.train()

    # conditioning projector (additive over future embedding)
    cond_proj = CondProjConcat(EMBED_DX).to(device)

    # if isinstance(state, dict) and "model" in state:
    #     model.load_state_dict(state["model"])
    #     if "cond_proj" in state:
    #         cond_proj.load_state_dict(state["cond_proj"])
    #         print("Loaded cond_proj from checkpoint")
    #     else:
    #         print("Warning: cond_proj not found in checkpoint")
    # else:
    #     # Legacy checkpoint format (only model state dict)
    #     model.load_state_dict(state)
    #     print("Warning: Legacy checkpoint format - cond_proj not loaded")

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

            # with amp.autocast('cuda', enabled=(device.type == 'cuda')):
            #     sigma, noised_tensor = sample_noise(feature_tensor)

            #     c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
            #     c_out  = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
            #     c_in   = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

            #     # precondition only future time steps
            #     result = noised_tensor.clone()
            #     c_in_broadcast = c_in[:, None, None, None]
            #     result[:, :, args.obs_len:, :] = c_in_broadcast * noised_tensor[:, :, args.obs_len:, :]

            #     # base embedding
            #     embedded = embed_features(result, sigma)

            #     # direction one-hot from future theta (teacher forcing during training)
            #     theta_future = feature_tensor[:, :, args.obs_len:, 2]
            #     y_onehot = direction_onehot_from_theta(theta_future, turn_thresh_deg=args.turn_thresh_deg)
            #     c_embed = cond_proj(y_onehot)  # [B,A,EMBED_DX]
            #     # expand to future horizon and add only on future steps
            #     B, A, Tp = theta_future.shape
            #     cond_time = c_embed.unsqueeze(2).expand(B, A, Tp, EMBED_DX)

            #     p_uncond = 0.1  # 10% of training examples are unconditioned; tune between 0.05..0.3

            #     # c_embed: [B, A, EMBED_DX]  (from cond_proj(y_onehot))
            #     # cond_time: [B, A, Tp, EMBED_DX] (after unsqueeze/expand)

            #     if p_uncond > 0:
            #         # draw a mask per batch element: 1 = keep condition, 0 = drop
            #         keep_mask = (torch.rand((B,), device=c_embed.device) > p_uncond).float()  # [B]
            #         # reshape to [B,1,1,1] so it broadcasts over A,Tp,EMBED_DX
            #         keep_mask = keep_mask.view(B, 1, 1, 1)
            #         cond_time = cond_time * keep_mask  # zero-out condition for dropped samples



            #     embedded[:, :, args.obs_len:, :] = embedded[:, :, args.obs_len:, :] + cond_time

            #     model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, args.obs_len:, :]
            with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                sigma, noised_tensor = sample_noise(feature_tensor)

                c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
                c_out  = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
                c_in   = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

                # precondition only future time steps
                result = noised_tensor.clone()
                c_in_broadcast = c_in[:, None, None, None]
                result[:, :, args.obs_len:, :] = c_in_broadcast * noised_tensor[:, :, args.obs_len:, :]

                # base embedding from features (shape [B,A,T,EMBED_DX])
                embedded = embed_features(result, sigma)

                # direction one-hot from future theta (teacher forcing during training)
                theta_future = feature_tensor[:, :, args.obs_len:, 2]
                y_onehot = direction_onehot_from_theta(theta_future, turn_thresh_deg=args.turn_thresh_deg)
                c_embed = cond_proj(y_onehot)  # [B,A,EMBED_DX]

                # expand to full time horizon (T) but only filled for future steps
                B, A, Tp = theta_future.shape
                T = embedded.shape[2]  # obs_len + pred_len
                # cond_full: [B,A,T,EMBED_DX], zeros for past
                cond_full = torch.zeros((B, A, T, EMBED_DX), device=embedded.device, dtype=embedded.dtype)
                cond_time = c_embed.unsqueeze(2).expand(B, A, Tp, EMBED_DX)  # [B,A,T_future,EMBED_DX]
                cond_full[:, :, args.obs_len:, :] = cond_time

                # classifier-free: with prob p_uncond replace c_embed with null token
                p_uncond = 0.1
                if p_uncond > 0:
                    # for each sample in batch, decide to drop condition
                    keep_mask = (torch.rand((B,), device=embedded.device) > p_uncond).float()  # [B]
                    # expand to [B,1,1,1] for broadcasting
                    keep_mask_b = keep_mask.view(B, 1, 1, 1)
                    # where keep_mask==0: replace cond_full with learned null token (broadcasted)
                    # build null tensor: [1,1,T,EMBED_DX]  -> broadcast to [B,A,T,EMBED_DX]
                    null = cond_proj.null_token.view(1, 1, 1, EMBED_DX).expand(B, A, T, EMBED_DX)
                    cond_full = cond_full * keep_mask_b + null * (1.0 - keep_mask_b)

                # CONCATENATE along feature dimension
                # embedded: [B,A,T,EMBED_DX]
                # cond_full: [B,A,T,EMBED_DX]
                embedded_concat = torch.cat([embedded, cond_full], dim=-1)  # [B,A,T,EMBED_DX*2]

                # forward through model (remember model expects embed_dim*2 in FeatureMLP)
                model_out = model(embedded_concat, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, args.obs_len:, :]
                
                
                gt_pred   = feature_tensor[:, :, args.obs_len:, :]
                mask_pred = feature_mask[:, :, args.obs_len:]
                valid_mask = mask_pred.unsqueeze(-1).expand_as(gt_pred)
                # out_of_bounds = (valid_mask == 0)  # Identify out-of-bounds predictions
                recon = model_out * c_out[:, None, None, None] + noised_tensor[:, :, args.obs_len:, :] * c_skip[:, None, None, None]

                squared_diff = (recon - gt_pred) ** 2
                masked_squared_diff = squared_diff * valid_mask.float()
                loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)

                weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
                diffusion_loss = (loss_per_batch * weight).mean()
                
                
                road_boundary_loss = road_boundary_mask_loss(
                    predictions=recon,
                    roadgraph_tensor=roadgraph_tensor,
                    roadgraph_mask=roadgraph_mask,
                    scene_std=scene_stds[0],  # Use first sample's std for all in batch
                    boundary_threshold=args.boundary_threshold,
                    loss_weight=args.road_boundary_loss_weight
                )
                
                # Combine losses
                loss = diffusion_loss + road_boundary_loss

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
            writer.add_scalar("train/diffusion_loss", float(diffusion_loss.item()), global_step)
            writer.add_scalar("train/road_boundary_loss", float(road_boundary_loss.item()), global_step)
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
                Flag_cond = args.condition

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
    p = argparse.ArgumentParser(description="Train diffusion model with maneuver conditioning (L/R/S)")
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
    p.add_argument('--turn_thresh_deg', type=float, default=15.0, help='Angle threshold to classify L/R/S')

    p.add_argument('--val_every', type=int, default=20)

    p.add_argument('--model_type', type=str, default='diffusion_cond', help='Model type for directory structure')
    
    # Conditioning guidance for eval
    p.add_argument('--cond_scale', type=float, default=2.0, help='Classifier-free guidance scale during eval')

    # Road boundary mask loss parameters
    p.add_argument('--road_boundary_loss_weight', type=float, default=1, help='Weight for road boundary mask loss')
    p.add_argument('--boundary_threshold', type=float, default=0, help='Distance threshold for road boundaries (meters)')
    
    # Evaluation options
    p.add_argument('--eval_only', action='store_true', help='Run evaluation only (no training)')
    p.add_argument('--ckpt_path', type=str, default=None, help='Path to model checkpoint for eval')
    p.add_argument('--eval_save_dir', type=str, default=None, help='Where to save eval plots')

    p.add_argument('--condition', type=bool, default=False, help='Condition Flag')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Denoiser().to(device)
        cond_proj = CondProjConcat(EMBED_DX).to(device)
      
        
        # Auto-find latest checkpoint if not provided
        if not args.ckpt_path:
            latest_ckpt = find_latest_checkpoint(args.model_type)
            if latest_ckpt:
                args.ckpt_path = latest_ckpt
                print(f"Auto-found latest checkpoint: {args.ckpt_path}")
            else:
                print("Warning: No checkpoint found. Evaluating with randomly initialized model.")
        
        # Load checkpoint if available
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

        # Auto-create eval save directory if not provided
        if not args.eval_save_dir or args.eval_save_dir == './eval_outputs':
            args.eval_save_dir = create_eval_save_dir(args.model_type)
            print(f"Auto-created eval save directory: {args.eval_save_dir}")
        
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
            Flag_cond = args.condition,
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
