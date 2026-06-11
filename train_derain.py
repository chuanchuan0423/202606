import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class AdaIRModel(pl.LightningModule):
    def __init__(self, opt, eval_interval):
        super().__init__()
        self.opt = opt
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.eval_datasets()
        self.eval_interval = eval_interval

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
        scheduler.get_last_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self, unused=None):
        if (self.current_epoch + 1) % self.eval_interval == 0:
            self.test_Derain_Dehaze(self.derain_set, task="derain")

    def eval_datasets(self):
        derain_splits = ["Rain100L/"]
        derain_base_path = self.opt.derain_path
        for name in derain_splits:
            self.opt.derain_path = os.path.join(derain_base_path, name)
            self.derain_set = DerainDehazeDataset(self.opt, addnoise=False, sigma=15)

    def test_Derain_Dehaze(self, dataset, task="derain"):
        dataset.set_dataset(task)
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

                restored = self.forward(degrad_patch)[:, :, :h, :w]
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

            self.log("psnr derain", psnr.avg)
            self.log("SSIM derain", ssim.avg)
            print("PSNR_derain: %.2f, SSIM_derain: %.4f" % (psnr.avg, ssim.avg))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')

    parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size to use per GPU")
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

    parser.add_argument('--de_type', nargs='+', default=['derain'],
                        help='which type of degradations is training and testing for.')

    parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

    parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where index txt files save.')
    parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                        help='where training images of deraining saves.')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/Derain/", help='checkpoint save path')
    parser.add_argument("--ckpt_dir", type=str, default="AdaIR-Derain", help="Name of the Directory where the checkpoint is to be saved")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--save_every_n_epochs", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--monitor_metric", type=str, default="psnr derain", help="Metric to monitor for best checkpoint")

    opt = parser.parse_args()

    path = opt.ckpt_dir + '_model'
    if not os.path.exists(path):
        os.makedirs(path)
    os.system('cp net/model.py ' + path)

    torch.set_float32_matmul_precision('high')

    logger = CSVLogger(save_dir="AdaIR-Derain/")

    trainset = AdaIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=opt.save_every_n_epochs,
        save_top_k=-1,
        filename="epoch_{epoch:03d}"
    )
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        monitor=opt.monitor_metric,
        mode="max",
        save_top_k=1,
        filename="best",
        every_n_epochs=opt.eval_interval
    )
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = AdaIRModel(opt, eval_interval=opt.eval_interval)
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true", logger=logger,
                         callbacks=[checkpoint_callback, best_checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
