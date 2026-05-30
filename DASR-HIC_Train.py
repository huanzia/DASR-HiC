#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Models.DiCARN_model import Generator
from Utils.SSIM import ssim


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(PROJECT_ROOT, "data", "patch")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "train")
SCORE_DIR = os.path.join(PROJECT_ROOT, "score_tracker")


def mkdir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for DASR-HiC training."""
    parser = argparse.ArgumentParser(
        description="Train DASR-HiC with distance-aware supervision."
    )

    parser.add_argument("-c", "--cell_line", type=str, required=True)
    parser.add_argument("-hr", "--high_res", type=str, required=True)
    parser.add_argument("-lr", "--low_res", type=str, required=True)

    parser.add_argument("-epochs", "--epochs", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--bound", type=int, default=201)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--pool_type", type=str, default="max")

    parser.add_argument(
        "--loss_mode",
        type=str,
        default="dist_far_strong",
        choices=["dist_only", "dist_far_strong"],
        help=(
            "Training variant. Use dist_only for transfer-oriented training "
            "or dist_far_strong for structure-oriented long-range recovery."
        ),
    )

    parser.add_argument("--base_lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument(
        "--lambda_dist",
        type=float,
        default=None,
        help="Override the default distance-aware loss weight.",
    )

    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=1e-8)

    return parser


def set_seed(seed: int = 42) -> None:
    """Set random seeds for stable and reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN behavior improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loss_config(loss_mode: str) -> Dict[str, float]:
    """Return the distance-aware loss configuration."""
    configs = {
        "dist_only": {
            "lambda_dist": 0.5,
            "w_near": 1.0,
            "w_mid": 1.2,
            "w_far": 1.5,
            "tag": "DASR_HiC_DistOnly",
        },
        "dist_far_strong": {
            "lambda_dist": 0.5,
            "w_near": 1.0,
            "w_mid": 1.2,
            "w_far": 1.8,
            "tag": "DASR_HiC_DistFarStrong",
        },
    }
    return configs[loss_mode]


def adjust_learning_rate(base_lr: float, epoch: int) -> float:
    """Apply step decay at epoch 30 and 60."""
    return base_lr * (0.1 ** (epoch // 30))


def build_distance_maps(
    patch_size: int,
    device: torch.device,
    w_near: float,
    w_mid: float,
    w_far: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the upper-triangle mask and distance-aware weight map.

    The contact map is divided into near-, middle-, and far-distance regions
    according to the offset from the main diagonal. Only the upper triangle
    excluding the diagonal is used to avoid duplicated symmetric entries.
    """
    ii, jj = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    offset = torch.abs(ii - jj).float()

    upper_mask = (ii < jj).float()

    max_offset = max(patch_size - 1, 1)
    near_thr = max_offset / 3.0
    mid_thr = 2.0 * max_offset / 3.0

    dist_weight = torch.ones_like(offset)
    dist_weight[offset <= near_thr] = w_near
    dist_weight[(offset > near_thr) & (offset <= mid_thr)] = w_mid
    dist_weight[offset > mid_thr] = w_far

    return upper_mask, dist_weight


def weighted_mse_upper(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    upper_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute distance-aware weighted MSE on the upper-triangle region.

    Args:
        pred: Predicted high-resolution patches with shape [B, C, H, W].
        target: Ground-truth high-resolution patches with shape [B, C, H, W].
        weight_map: Distance-aware weight map with shape [H, W].
        upper_mask: Upper-triangle mask with shape [H, W].
        eps: Numerical stability term.

    Returns:
        Batch-averaged weighted MSE loss.
    """
    diff2 = (pred - target) ** 2
    weight = (weight_map * upper_mask).unsqueeze(0).unsqueeze(0)

    numerator = (diff2 * weight).sum(dim=(1, 2, 3))
    denominator = weight.sum() + eps

    return (numerator / denominator).mean()


def load_npz_dataset(path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a processed Hi-C patch dataset from an NPZ file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=True)
    x = torch.tensor(data["data"], dtype=torch.float32)
    y = torch.tensor(data["target"], dtype=torch.float32)
    inds = torch.tensor(data["inds"], dtype=torch.long)

    return x, y, inds


def main() -> None:
    """Run DASR-HiC training."""
    args = build_parser().parse_args(sys.argv[1:])

    set_seed(args.seed)

    config = get_loss_config(args.loss_mode)
    if args.lambda_dist is not None:
        config["lambda_dist"] = args.lambda_dist

    print("\n==============================")
    print(f"LOSS MODE: {args.loss_mode}")
    print(f"LOSS CONFIG: {config}")
    print(f"SEED: {args.seed}")
    print("==============================\n")

    pool_str = "nonpool" if args.scale == 1 else f"{args.pool_type}pool{args.scale}"
    resos = f"{args.high_res}{args.low_res}"

    patch_dir = os.path.join(PATCH_DIR, args.cell_line)
    ckpt_dir = os.path.join(OUTPUT_DIR, args.cell_line)
    log_dir = os.path.join(LOG_DIR, args.cell_line)
    score_dir = os.path.join(SCORE_DIR, args.cell_line)

    mkdir(ckpt_dir)
    mkdir(log_dir)
    mkdir(score_dir)

    train_file = os.path.join(
        patch_dir,
        f"hicarn_{resos}_c{args.chunk}_s{args.stride}_b{args.bound}_{pool_str}_train.npz",
    )
    valid_file = os.path.join(
        patch_dir,
        f"hicarn_{resos}_c{args.chunk}_s{args.stride}_b{args.bound}_{pool_str}_valid.npz",
    )

    print("Train file:", train_file)
    print("Valid file:", valid_file)

    use_cuda = torch.cuda.is_available()
    device = torch.device(
        f"cuda:{args.cuda}"
        if use_cuda and 0 <= args.cuda < torch.cuda.device_count()
        else "cpu"
    )

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    start_time = time.time()

    train_x, train_y, train_inds = load_npz_dataset(train_file)
    valid_x, valid_y, valid_inds = load_npz_dataset(valid_file)

    train_set = TensorDataset(train_x, train_y, train_inds)
    valid_set = TensorDataset(valid_x, valid_y, valid_inds)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    patch_size = int(train_x.shape[-1])
    upper_mask, dist_weight = build_distance_maps(
        patch_size=patch_size,
        device=device,
        w_near=config["w_near"],
        w_mid=config["w_mid"],
        w_far=config["w_far"],
    )

    net_g = Generator(num_channels=64).to(device)
    rec_criterion = nn.MSELoss()
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.base_lr)

    date_str = time.strftime("%m_%d_%H_%M")
    model_tag = config["tag"]

    log_txt = os.path.join(
        log_dir,
        f"{date_str}_{model_tag}_{args.cell_line}_seed{args.seed}.txt",
    )
    log_csv = os.path.join(
        log_dir,
        f"{date_str}_{model_tag}_{args.cell_line}_seed{args.seed}.csv",
    )

    with open(log_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_total",
                "train_rec",
                "train_dist",
                "valid_ssim",
                "valid_psnr",
                "valid_mse",
                "valid_mae",
                "skipped",
            ]
        )

    best_ssim = -1.0
    skipped_total = 0

    valid_ssim_scores = []
    valid_psnr_scores = []
    valid_mse_scores = []
    valid_mae_scores = []

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args.base_lr, epoch)
        for param_group in optimizer_g.param_groups:
            param_group["lr"] = lr

        net_g.train()

        train_result = {
            "nsamples": 0,
            "total": 0.0,
            "rec": 0.0,
            "dist": 0.0,
            "skipped": 0,
        }

        train_bar = tqdm(train_loader, desc=f"[Train {epoch}/{args.epochs}]")

        for data, target, _ in train_bar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            batch_size = data.size(0)

            optimizer_g.zero_grad(set_to_none=True)
            pred = net_g(data)

            rec_loss = rec_criterion(pred, target)
            dist_loss = weighted_mse_upper(
                pred=pred,
                target=target,
                weight_map=dist_weight,
                upper_mask=upper_mask,
                eps=args.eps,
            )

            total_loss = (
                args.lambda_rec * rec_loss
                + config["lambda_dist"] * dist_loss
            )

            if not torch.isfinite(total_loss):
                train_result["skipped"] += 1
                skipped_total += 1
                continue

            total_loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), args.grad_clip)

            optimizer_g.step()

            train_result["nsamples"] += batch_size
            train_result["total"] += total_loss.item() * batch_size
            train_result["rec"] += rec_loss.item() * batch_size
            train_result["dist"] += dist_loss.item() * batch_size

            avg_total = train_result["total"] / max(train_result["nsamples"], 1)
            avg_rec = train_result["rec"] / max(train_result["nsamples"], 1)
            avg_dist = train_result["dist"] / max(train_result["nsamples"], 1)

            train_bar.set_description(
                f"[Train {epoch}/{args.epochs}] "
                f"Total: {avg_total:.6f} | Rec: {avg_rec:.6f} | Dist: {avg_dist:.6f}"
            )

        net_g.eval()
        valid_result = {
            "nsamples": 0,
            "mse": 0.0,
            "mae": 0.0,
            "ssims": 0.0,
        }

        valid_bar = tqdm(valid_loader, desc=f"[Valid {epoch}/{args.epochs}]")

        with torch.no_grad():
            for val_x, val_y, _ in valid_bar:
                val_x = val_x.to(device, non_blocking=True)
                val_y = val_y.to(device, non_blocking=True)
                batch_size = val_x.size(0)

                sr = net_g(val_x)

                batch_mse = ((sr - val_y) ** 2).mean()
                batch_mae = (sr - val_y).abs().mean()
                batch_ssim = ssim(sr[:, 0:1, :, :], val_y[:, 0:1, :, :])

                valid_result["nsamples"] += batch_size
                valid_result["mse"] += batch_mse.item() * batch_size
                valid_result["mae"] += batch_mae.item() * batch_size
                valid_result["ssims"] += batch_ssim.item() * batch_size

                cur_mse = valid_result["mse"] / max(valid_result["nsamples"], 1)
                cur_ssim = valid_result["ssims"] / max(valid_result["nsamples"], 1)
                cur_psnr = 10 * math.log10(1.0 / max(cur_mse, 1e-12))

                valid_bar.set_description(
                    f"[Valid {epoch}/{args.epochs}] "
                    f"PSNR: {cur_psnr:.4f} dB SSIM: {cur_ssim:.4f}"
                )

        now_mse = valid_result["mse"] / max(valid_result["nsamples"], 1)
        now_mae = valid_result["mae"] / max(valid_result["nsamples"], 1)
        now_ssim = valid_result["ssims"] / max(valid_result["nsamples"], 1)
        now_psnr = 10 * math.log10(1.0 / max(now_mse, 1e-12))

        valid_ssim_scores.append(now_ssim)
        valid_psnr_scores.append(now_psnr)
        valid_mse_scores.append(now_mse)
        valid_mae_scores.append(now_mae)

        train_total_avg = train_result["total"] / max(train_result["nsamples"], 1)
        train_rec_avg = train_result["rec"] / max(train_result["nsamples"], 1)
        train_dist_avg = train_result["dist"] / max(train_result["nsamples"], 1)

        line = (
            f"Epoch {epoch:03d}, lr={lr:.8f}, "
            f"train_total={train_total_avg:.8f}, "
            f"train_rec={train_rec_avg:.8f}, "
            f"train_dist={train_dist_avg:.8f}, "
            f"valid_ssim={now_ssim:.8f}, "
            f"valid_psnr={now_psnr:.8f}, "
            f"valid_mse={now_mse:.10f}, "
            f"valid_mae={now_mae:.10f}, "
            f"skipped={train_result['skipped']}"
        )
        print(line)

        with open(log_txt, "a", encoding="utf-8") as ftxt:
            ftxt.write(line + "\n")

        with open(log_csv, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(
                [
                    epoch,
                    lr,
                    train_total_avg,
                    train_rec_avg,
                    train_dist_avg,
                    now_ssim,
                    now_psnr,
                    now_mse,
                    now_mae,
                    train_result["skipped"],
                ]
            )

        if now_ssim > best_ssim:
            best_ssim = now_ssim
            best_ckpt = (
                f"{date_str}_bestg_{resos}_c{args.chunk}_s{args.stride}_b{args.bound}_"
                f"{pool_str}_{model_tag}_{args.cell_line}_seed{args.seed}_stable.pth"
            )
            best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)
            torch.save(net_g.state_dict(), best_ckpt_path)
            print(f"Now, best SSIM is {best_ssim:.8f}")
            print("Saved best checkpoint to:", best_ckpt_path)

    final_ckpt = (
        f"{date_str}_finalg_{resos}_c{args.chunk}_s{args.stride}_b{args.bound}_"
        f"{pool_str}_{model_tag}_{args.cell_line}_seed{args.seed}_stable.pth"
    )
    final_ckpt_path = os.path.join(ckpt_dir, final_ckpt)
    torch.save(net_g.state_dict(), final_ckpt_path)

    np.savetxt(
        os.path.join(
            score_dir,
            f"valid_ssim_scores_{model_tag}_{args.cell_line}_seed{args.seed}.txt",
        ),
        np.array(valid_ssim_scores),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            score_dir,
            f"valid_psnr_scores_{model_tag}_{args.cell_line}_seed{args.seed}.txt",
        ),
        np.array(valid_psnr_scores),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            score_dir,
            f"valid_mse_scores_{model_tag}_{args.cell_line}_seed{args.seed}.txt",
        ),
        np.array(valid_mse_scores),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            score_dir,
            f"valid_mae_scores_{model_tag}_{args.cell_line}_seed{args.seed}.txt",
        ),
        np.array(valid_mae_scores),
        delimiter=",",
    )

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nCheckpoint dir:", ckpt_dir)
    print("Best SSIM:", best_ssim)
    print("Total skipped batches:", skipped_total)
    print("Final checkpoint:", final_ckpt_path)
    print("Log txt:", log_txt)
    print("Log csv:", log_csv)
    print(
        "\nTotal training time: {:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )
    )


if __name__ == "__main__":
    main()
