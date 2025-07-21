import cv2
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from os.path import join
from tools import get_padding_size
from networks.roma.roma import RoMa,RegressionMatcher
import os
from tqdm import tqdm
import rasterio
import rasterio.errors
import warnings

from torch.utils.data import Dataset,DataLoader
from typing import Tuple,List


warnings.filterwarnings("ignore")

def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized

def preprocess(image: torch.Tensor, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.numpy()
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = np.stack([image] * 3,axis=0)
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale

def match_one_pair(model:RegressionMatcher,image0,image1):
    image0, scale0 = preprocess(image0,grayscale=True)
    image1, scale1 = preprocess(image1,grayscale=True)

    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]

    b_ids, mconf, kpts0, kpts1 = None, None, None, None
    # data = dict(color0=image0, color1=image1, image0=image0, image1=image1)

    width, height = 672, 672

    orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0, width, height)
    orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1, width, height)
    image0_ = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
    image1_ = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    dense_matches, dense_certainty = model.match(image0_, image1_)
    sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 5000)

    height0, width0 = image0_.shape[-2:]
    height1, width1 = image1_.shape[-2:]

    kpts0 = sparse_matches[:, :2]
    kpts0 = torch.stack((
        width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
    kpts1 = sparse_matches[:, 2:]
    kpts1 = torch.stack((
        width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

    # before padding
    kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
    kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
    mask_ = (kpts0[:, 0] > 0) & \
            (kpts0[:, 1] > 0) & \
            (kpts1[:, 0] > 0) & \
            (kpts1[:, 1] > 0)
    mask_ = mask_ & \
            (kpts0[:, 0] <= (orig_width0 - 1)) & \
            (kpts1[:, 0] <= (orig_width1 - 1)) & \
            (kpts0[:, 1] <= (orig_height0 - 1)) & \
            (kpts1[:, 1] <= (orig_height1 - 1))

    # mconf = mconf[mask_]
    kpts0 = kpts0[mask_].cpu().numpy()
    kpts1 = kpts1[mask_].cpu().numpy()

    return kpts0,kpts1

def match_img(model:RegressionMatcher,img1:np.ndarray,img2:np.ndarray):
    H,W = img1.shape[:2]
    size = 672
    step = size
    line_num = max((H - size) // step,1)
    samp_num = max((W - size) // step,1)
    line_step = (H - size) // line_num
    samp_step = (W - size) // samp_num
    
    lines = np.arange(0,H - size,line_step)
    samps = np.arange(0,W - size,samp_step)

    pbar = tqdm(total=len(lines) * len(samps))

    kpts1_total = []
    kpts2_total = []
    for line in lines:
        for samp in samps:
            img1_crop = torch.from_numpy(img1[line:line+size,samp:samp+size])
            img2_crop = torch.from_numpy(img2[line:line+size,samp:samp+size])
            kpts1,kpts2 = match_one_pair(model,img1_crop,img2_crop)
            kpts1 += [line,samp]
            kpts2 += [line,samp]
            kpts1_total.append(kpts1)
            kpts2_total.append(kpts2)
            pbar.update(1)
    
    return np.concatenate(kpts1_total,axis=0),np.concatenate(kpts2_total,axis=0)

def get_residuals(model:RegressionMatcher,imgs:List[np.ndarray]):
    H,W = 3000,3000
    img_num = len(imgs)
    residuals = [np.full((H,W),np.nan,dtype=np.float32) for i in range(img_num)]
    counts = [np.full((H,W),0,dtype=np.float32) for i in range(img_num)]
    for i in range(img_num-1):
        for j in range(i+1,img_num):
            print(f"matching img{i+1} and img{j+1}")
            img_i = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            img_j = cv2.imread(imgs[j],cv2.IMREAD_GRAYSCALE)
            kpts_i,kpts_j = match_img(model,img_i,img_j)
            dis = np.linalg.norm(kpts_i - kpts_j,axis=-1)
            idx_i = np.round(kpts_i).astype(int)
            idx_j = np.round(kpts_j).astype(int)
            residuals[i][idx_i[:,0],idx_i[:,1]] += dis
            residuals[j][idx_j[:,0],idx_j[:,1]] += dis
            counts[i][idx_i[:,0],idx_i[:,1]] += 1.
            counts[j][idx_j[:,0],idx_j[:,1]] += 1.
            # residuals[i][idx_i[:,0],idx_i[:,1]] = np.fmin(residuals[i][idx_i[:,0],idx_i[:,1]],dis)
            # residuals[j][idx_j[:,0],idx_j[:,1]] = np.fmin(residuals[j][idx_j[:,0],idx_j[:,1]],dis)
    for i in range(img_num):
        count = counts[i]
        valid_mask = count > 0
        residuals[i][valid_mask] /= count[valid_mask]

    return residuals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()

    root = args.root

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RoMa(img_size=[672])
    checkpoints_path = './weights/gim_roma_100h.ckpt'
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    print("model loaded!")

    folders = os.listdir(root)

    for folder in folders:
        print(f"processing {folder}")
        path = os.path.join(root,folder)
        names = [i.split('.png')[0] for i in os.listdir(path) if 'png' in i]
        img_paths = [os.path.join(path,f'{name}.png') for name in names]
        residuals = get_residuals(model,img_paths)

        for i,residual in enumerate(residuals):
            np.save(os.path.join(path,f'{names[i]}_res.npy'),residual)

            

