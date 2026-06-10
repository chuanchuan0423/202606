import os
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import StarIR


class StarIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = StarIR()
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


def test_Derain(net, dataset, output_path):
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset("derain")
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    factor = 32

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            b, c, h, w = degrad_patch.shape
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            degrad_patch = torch.nn.functional.pad(degrad_patch, (0, w_n, 0, h_n), mode='reflect')

            restored = net(degrad_patch)[:, :, :h, :w]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

    print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--derain_path', type=str, default="data/test/derain/Rain100L/",
                        help='path to test rainy images')
    parser.add_argument('--output_path', type=str, default="results/derain/",
                        help='path to save restored images')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='full path to checkpoint file (.ckpt)')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    print("Loading checkpoint: {}".format(testopt.ckpt_path))
    net = StarIRModel.load_from_checkpoint(testopt.ckpt_path).cuda()
    net.eval()

    derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)

    print('Start testing derain...')
    test_Derain(net, derain_set, testopt.output_path)
