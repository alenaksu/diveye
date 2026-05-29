"""
LU2Net Training Script
======================
Trains (or fine-tunes) LU2Net on a paired underwater image dataset.

Reference
---------
  Yang et al., "LU2Net: A Lightweight Network for Real-time Underwater Image Enhancement"
  arXiv:2406.14973v1  https://arxiv.org/html/2406.14973v1

  Loss function (Section III-D, Eq. 1):
    L_total = l_RGB + l_LAB + l_LCH + l_SSIM + l_VGG
  where l_RGB, l_LAB, l_LCH are MSE in the respective color spaces,
  l_SSIM = 1 - SSIM, and l_VGG is a VGG19 perceptual loss.

Dataset layout expected
-----------------------
  <data-dir>/
      input/   -- degraded/raw underwater images
      GT/      -- ground-truth / reference images

Both folders must contain identically-named image files.

LSUI dataset (recommended)
---------------------------
  Download from Kaggle:
    https://www.kaggle.com/api/v1/datasets/download/noureldin199/lsui-large-scale-underwater-image-dataset
  Unzip; the folder structure already matches input/ GT/.

Usage
-----
  # Fresh training
  python train.py --data-dir /path/to/lsui

  # Resume / fine-tune from LightUNet_170.pth (epoch 170)
  python train.py --data-dir /path/to/lsui --resume LightUNet_170.pth

  # Fine-tune with lower LR
  python train.py --data-dir /path/to/lsui --resume LightUNet_170.pth \\
                  --lr 1e-5 --epochs 200

  # After training, export to ONNX:
  python export_onnx.py --weights runs/<run>/best.pth
"""

import argparse
import os
import sys
import time
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Add the scripts directory to sys.path so sibling modules resolve
sys.path.insert(0, os.path.dirname(__file__))

from LU2Net import LU2Net
from utility.data import LSUIDataset
from utility.ptcolor import rgb2lab, rgb2lch
from loss.VGG19_PercepLoss import VGG19_PercepLoss
import pytorch_ssim


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train LU2Net for underwater image enhancement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="Root dataset directory containing input/ and GT/ folders")
    p.add_argument("--val-split", type=float, default=0.05,
                   help="Fraction of training data to use as validation")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Random crop size (must be divisible by 4)")
    p.add_argument("--epochs", type=int, default=300,
                   help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Initial learning rate")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum LR for cosine schedule")
    p.add_argument("--no-vgg", action="store_true",
                   help="Disable VGG19 perceptual loss (faster, less memory)")
    p.add_argument("--resume", default=None,
                   help="Path to .pth checkpoint to resume/fine-tune from")
    p.add_argument("--run-dir", default="runs",
                   help="Directory for checkpoints and TensorBoard logs")
    p.add_argument("--run-name", default=None,
                   help="Sub-folder name inside --run-dir (auto-generated if omitted)")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save a checkpoint every N epochs")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count")
    p.add_argument("--device", default=None,
                   help="Torch device (e.g. cuda, cuda:1, cpu). Auto-detected if omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_dir(base, name=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = name or f"lu2net_{ts}"
    path = Path(base) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load weights from a checkpoint.  Returns (start_epoch, best_psnr)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "model" in state:
            model.load_state_dict(state["model"])
        else:
            # Raw state_dict (e.g. LightUNet_170.pth)
            model.load_state_dict(state)
        start_epoch = state.get("epoch", 0) + 1 if isinstance(state, dict) and "epoch" in state else 0
        best_psnr   = state.get("best_psnr", 0.0)  if isinstance(state, dict) else 0.0
        if optimizer is not None and "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass  # LR changed — ignore optimizer state
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(state)}")
    return start_epoch, best_psnr


def save_checkpoint(path, model, optimizer, epoch, best_psnr, is_best, run_dir):
    ckpt = {
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "best_psnr":  best_psnr,
    }
    torch.save(ckpt, path)
    if is_best:
        shutil.copy(path, run_dir / "best.pth")


def psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] device = {device}")

    # ------------------------------------------------------------------
    # Dataset & DataLoaders
    # ------------------------------------------------------------------
    assert args.crop_size % 4 == 0, "--crop-size must be divisible by 4"

    full_ds = LSUIDataset(args.data_dir, training=True, crop_size=args.crop_size)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Val dataset: no crop/flip — use full images resized to multiple of 4
    val_ds_full = LSUIDataset(args.data_dir, training=False, crop_size=args.crop_size)
    # Use same indices as the split
    val_ds_full = torch.utils.data.Subset(val_ds_full, val_ds.indices)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds_full, batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)

    print(f"[train] {n_train} train / {n_val} val images")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = LU2Net().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] LU2Net parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB fp32)")

    # ------------------------------------------------------------------
    # Losses  (paper Eq. 1: L_total = l_RGB + l_LAB + l_LCH + l_SSIM + l_VGG)
    # ------------------------------------------------------------------
    criterion_ssim = pytorch_ssim.SSIM(window_size=11).to(device)
    criterion_percep = None
    if not args.no_vgg:
        try:
            criterion_percep = VGG19_PercepLoss().to(device)
            print("[train] VGG19 perceptual loss enabled")
        except Exception as e:
            print(f"[train] WARNING: Could not load VGG19 ({e}). Perceptual loss disabled.")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    run_dir    = make_run_dir(args.run_dir, args.run_name)
    start_epoch = 0
    best_psnr   = 0.0

    if args.resume:
        print(f"[train] Loading checkpoint: {args.resume}")
        start_epoch, best_psnr = load_checkpoint(
            args.resume, model, optimizer, device=str(device)
        )
        # Fast-forward the scheduler to match
        for _ in range(start_epoch):
            scheduler.step()
        print(f"[train] Resuming from epoch {start_epoch}, best PSNR {best_psnr:.2f} dB")

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(f"[train] Logs & checkpoints → {run_dir}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (raw, gt, _) in enumerate(train_loader):
            raw = raw.to(device, non_blocking=True)
            gt  = gt.to(device,  non_blocking=True)

            pred = model(raw)
            pred = torch.clamp(pred, 0.0, 1.0)

            # l_RGB: MSE in RGB space
            loss_rgb  = nn.functional.mse_loss(pred, gt)
            # l_LAB: MSE in LAB space
            loss_lab  = nn.functional.mse_loss(rgb2lab(pred), rgb2lab(gt))
            # l_LCH: MSE in LCH space
            loss_lch  = nn.functional.mse_loss(rgb2lch(pred), rgb2lch(gt))
            # l_SSIM: 1 - SSIM
            loss_ssim = 1.0 - criterion_ssim(pred, gt)
            loss = loss_rgb + loss_lab + loss_lch + loss_ssim

            if criterion_percep is not None:
                loss = loss + criterion_percep(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("train/loss", loss.item(), step)
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch+1:3d}/{args.epochs}  "
                      f"batch {batch_idx:4d}/{len(train_loader)}  "
                      f"loss {loss.item():.4f}  lr {lr_now:.2e}  "
                      f"({time.time()-t0:.1f}s)")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0

        with torch.no_grad():
            for i, (raw, gt, _) in enumerate(val_loader):
                raw = raw.to(device)
                gt  = gt.to(device)
                pred = torch.clamp(model(raw), 0.0, 1.0)
                val_psnr_sum += psnr(pred, gt)
                val_ssim_sum += criterion_ssim(pred, gt).item()

                # Log first 4 val images to TensorBoard
                if i < 4:
                    grid = vutils.make_grid(
                        torch.cat([raw, pred, gt], dim=0), nrow=1, normalize=True
                    )
                    writer.add_image(f"val/img_{i}_input-pred-gt", grid, epoch)

        avg_psnr = val_psnr_sum / len(val_loader)
        avg_ssim = val_ssim_sum / len(val_loader)
        writer.add_scalar("val/psnr", avg_psnr, epoch)
        writer.add_scalar("val/ssim", avg_ssim, epoch)

        is_best  = avg_psnr > best_psnr
        best_psnr = max(avg_psnr, best_psnr)

        print(f"[epoch {epoch+1:3d}] loss={avg_loss:.4f}  "
              f"PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}  "
              f"{'*** best ***' if is_best else ''}")

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_path = run_dir / f"epoch_{epoch+1:04d}.pth"
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_psnr, is_best, run_dir)

    writer.close()
    print(f"\n[train] Done. Best PSNR: {best_psnr:.2f} dB")
    print(f"[train] Best weights: {run_dir / 'best.pth'}")
    print(f"[train] Export with:  python export_onnx.py --weights {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
