import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F

import torch.nn as nn
import torch
from basicsr.utils.options import dict2str, parse

from skimage import img_as_ubyte
import cv2
import utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
from basicsr.models import create_model

import lpips
alex = lpips.LPIPS(net='alex').cuda()


parser = argparse.ArgumentParser(description='Dual-pixel Defocus Deblurring using StarIR')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/dpdd-dual/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/StarIR-DPDD-Dual.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
args = parser.parse_args()

####### Load yaml #######
yaml_file = './Options/DPDD-Dual.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
opt = parse(yaml_file, is_train=False)
opt['dist'] = False

opt['network_g']['basesize'] = 110

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

model_restoration = create_model(opt).net_g

checkpoint = torch.load(args.weights)
try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    # print('t')
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join('./Datasets/test/inputL', '*.png')))
filesR = natsorted(glob(os.path.join('./Datasets/test/inputR', '*.png')))
filesC = natsorted(glob(os.path.join('./Datasets/test/target', '*.png')))

indoor_labels  = np.load('./Datasets/test/indoor_labels.npy')
outdoor_labels = np.load('./Datasets/test/outdoor_labels.npy')
factor = 32
psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):

        imgL = np.float32(utils.load_img16(fileL))/65535.
        imgR = np.float32(utils.load_img16(fileR))/65535.
        imgC = np.float32(utils.load_img16(fileC))/65535.

        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0,3,1,2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0,3,1,2)

        input_ = torch.cat([patchL, patchR], 1).cuda()
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        restored = model_restoration(input_)[:, :, :h, :w]
        restored = torch.clamp(restored,0,1)
        pips.append(alex(patchC, restored, normalize=True).item())

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))
        if args.save_images:
            save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
            restored = np.uint16((restored*65535).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels-1], mae[indoor_labels-1], ssim[indoor_labels-1], pips[indoor_labels-1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels-1], mae[outdoor_labels-1], ssim[outdoor_labels-1], pips[outdoor_labels-1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor)))