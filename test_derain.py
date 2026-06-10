import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning.pytorch as pl

from net.model import AdaIR
from utils.dataset_utils import DerainDehazeDataset
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


def resolve_ckpt_path(ckpt_name):
    ckpt_path = Path(ckpt_name)
    if ckpt_path.is_file():
        return ckpt_path

    ckpt_path = Path("ckpt") / ckpt_name
    if ckpt_path.is_file():
        return ckpt_path

    raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_name} or {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test AdaIR on Rain100L deraining only.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id.")
    parser.add_argument(
        "--derain_path",
        type=str,
        default="data/test/derain/Rain100L/",
        help="Rain100L folder containing input/ and target/.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="adair-single-derain.ckpt",
        help="Checkpoint filename under ckpt/ or an absolute/relative checkpoint path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/derain/",
        help="Folder to save restored images.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda)

    input_dir = Path(args.derain_path) / "input"
    target_dir = Path(args.derain_path) / "target"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(
            f"Expected derain_path to contain input/ and target/: {args.derain_path}"
        )

    ckpt_path = resolve_ckpt_path(args.ckpt_name)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Derain test path: {args.derain_path}")
    print(f"Output path: {output_dir}")

    dataset = DerainDehazeDataset(args, task="derain", addnoise=False, sigma=15)
    loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AdaIRModel.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(loader):
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            restored = model(degrad_patch)
            temp_psnr, temp_ssim, count = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, count)
            ssim.update(temp_ssim, count)

            save_path = output_dir / f"{degraded_name[0]}.png"
            save_image_tensor(restored, os.fspath(save_path))

    print(f"Rain100L Derain PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")


if __name__ == "__main__":
    main()
