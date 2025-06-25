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
from networks.roma.roma import RoMa
import os
from tqdm import tqdm
import rasterio
import rasterio.errors
import warnings

from torch.utils.data import Dataset,DataLoader

warnings.filterwarnings("ignore")

class Image:
    def __init__(self, filepath: str):

        self.filepath = filepath
        self._src = None  # 存储 rasterio 数据源对象，延迟打开
        self._height = None
        self._width = None
        self._num_bands = None

        try:
            # 尝试打开文件以获取元数据，但不读取数据
            with rasterio.open(self.filepath) as src:
                self._height = src.height
                self._width = src.width
                self._num_bands = src.count
            print(f"影像 '{self.filepath}' 已成功初始化。")
            print(f"影像尺寸: H={self._height}, W={self._width}, 波段数={self._num_bands}")
        except FileNotFoundError:
            print(f"错误: 文件 '{filepath}' 未找到。请检查路径是否正确。")
            raise
        except rasterio.errors.RasterioIOError as e:
            print(f"错误: 无法打开文件 '{filepath}'。详细信息: {e}")
            raise
        except Exception as e:
            print(f"初始化时发生未知错误: {e}")
            raise

    def _open_source(self):
        """内部方法：惰性打开 rasterio 数据源。"""
        if self._src is None:
            self._src = rasterio.open(self.filepath)

    def _close_source(self):
        """内部方法：关闭 rasterio 数据源。"""
        if self._src is not None:
            self._src.close()
            self._src = None

    def get_size(self):
        """
        获取影像的高度和宽度。

        Returns:
            tuple[int, int]: (高度 H, 宽度 W)
        """
        return (self._height, self._width)

    def get_img(self, top_left_linesamp, bottom_right_linesamp) -> np.ndarray:
        """
        获取一个矩形窗口内的影像数据，并将其转换为全色图像。
        如果影像有多个波段，则计算平均值。

        Args:
            top_left_linesamp (tuple[int, int]): 矩形区域左上角的 (行号, 列号)。
                                                 行号和列号从0开始。
            bottom_right_linesamp (tuple[int, int]): 矩形区域右下角的 (行号, 列号)。
                                                     行号和列号从0开始。

        Returns:
            np.ndarray: (h, w) 形状的NumPy数组，表示矩形窗口内的全色图像。

        Raises:
            ValueError: 如果提供的坐标超出影像范围或不合法。
            rasterio.errors.RasterioIOError: 如果读取数据时发生错误。
        """
        r_start, c_start = top_left_linesamp
        r_end, c_end = bottom_right_linesamp

        # 校验坐标
        if not (0 <= r_start < self._height and 0 <= c_start < self._width and
                0 <= r_end < self._height and 0 <= c_end < self._width and
                r_start <= r_end and c_start <= c_end):
            raise ValueError(f"提供的矩形坐标 ({r_start},{c_start})-({r_end},{c_end}) 超出影像范围 "
                             f"(H={self._height}, W={self._width}) 或不合法。")

        window_height = r_end - r_start + 1
        window_width = c_end - c_start + 1

        # 定义读取窗口
        window = rasterio.windows.Window(col_off=c_start, row_off=r_start,
                                         width=window_width, height=window_height)

        self._open_source() # 确保数据源已打开
        try:
            if self._num_bands == 1:
                # 读取单波段数据
                img_data = self._src.read(1, window=window)
            else:
                # 读取所有波段数据，并计算平均值进行全色化
                multispectral_data = self._src.read(window=window)
                # 计算所有波段的平均值
                img_data = np.mean(multispectral_data, axis=0).astype(multispectral_data.dtype)
            
            # print(f"成功读取窗口 ({r_start},{c_start})-({r_end},{c_end})，返回形状: {img_data.shape}")
            return img_data
        except rasterio.errors.RasterioIOError as e:
            print(f"错误: 读取影像数据时发生错误。详细信息: {e}")
            raise
        # finally:
        #     # 每次读取后关闭数据源，避免长时间占用文件句柄，尤其在循环读取小块时
        #     # 如果需要频繁小块读取，可以考虑将_src的关闭移到类的析构函数中或由用户手动管理，
        #     # 但为了简单性和健壮性，这里选择每次操作后关闭。
        #     self._close_source()

    def __del__(self):
        """
        析构函数，确保在对象被销毁时关闭 rasterio 数据源。
        """
        self._close_source()
        print(f"影像 '{self.filepath}' 的资源已释放。")

class ImageDataset(Dataset):
    def __init__(self,image0:Image,image1:Image):
        super().__init__()
        self.image0 = image0
        self.image1 = image1
        H0,W0 = image0.get_size()
        H1,W1 = image1.get_size()
        if H0 != H1 or W0 != W1:
            print("错误：两张影像尺寸不匹配")
            exit()
        
        self.H,self.W = H0,W0
        
        self.size = 672
        self.step = self.size

        self.lines = np.arange(0,self.H - self.size,self.step)
        self.samps = np.arange(0,self.W - self.size,self.step)
        if self.lines[-1] < self.H - self.size - 1:
            self.lines = np.append(self.lines,self.H - self.size - 1)
        if self.samps[-1] < self.W - self.size - 1:
            self.samps = np.append(self.samps,self.W - self.size - 1)
        
        line_mesh,samp_mesh = np.meshgrid(self.lines,self.samps,indexing='ij')
        self.coords = np.stack([line_mesh.ravel(),samp_mesh.ravel()],axis=-1)
        
        self.patch_num = len(self.coords)
    
    def __len__(self):
        return self.patch_num
    
    def __getitem__(self, index):
        line,samp = self.coords[index]
        img0 = self.image0.get_img((line,samp),(line + self.size - 1,samp + self.size - 1))
        img1 = self.image1.get_img((line,samp),(line + self.size - 1,samp + self.size - 1))

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        line = torch.tensor([line])
        samp = torch.tensor([samp])

        return img0,img1,line,samp



def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


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


def match_one_pair(model:RoMa,image0,image1):
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
        b_ids = torch.where(mconf[None])[0]

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

        mconf = mconf[mask_]
        b_ids = b_ids[mask_]
        kpts0 = kpts0[mask_].cpu().numpy()
        kpts1 = kpts1[mask_].cpu().numpy()

        # if len(kpts0) == 0:
        #     kpts0 = torch.tensor([[orig_width0//2,orig_height0//2]],dtype=kpts0.dtype,device=kpts0.device)
        #     kpts1 = torch.tensor([[orig_width1//2,orig_height1//2]],dtype=kpts1.dtype,device=kpts1.device)

        # _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
        #                                 kpts1.cpu().detach().numpy(),
        #                                 cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
        #                                 confidence=0.999999, maxIters=10000)
        # mask = mask.ravel() > 0

        # kpts0 = kpts0[mask].cpu().numpy()
        # kpts1 = kpts1[mask].cpu().numpy()

        # with open(output_path,'w') as f:
        #     for kpt0,kpt1 in zip(kpts0,kpts1):
        #         f.write(f"{kpt0[1].item():.2f} {kpt0[0].item():.2f} {kpt1[1].item():.2f} {kpt1[0].item():.2f}\n")
        
        # return len(kpts0)
        return kpts0,kpts1


def match(model:RoMa,tif_path0:str,tif_path1:str,output_path:str,batch_size = 8):
    image0 = Image(tif_path0)
    image1 = Image(tif_path1)
    dataset = ImageDataset(image0,image1)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    patch_num = len(dataset)
    pbar = tqdm(total=patch_num)
    res0 = np.full((dataset.H,dataset.W),0,dtype=np.uint8)
    res1 = np.full((dataset.H,dataset.W),0,dtype=np.uint8)
    count = 0
    kpts0_total = []
    kpts1_total = []
    residuals_total = []

    for data in dataloader:
        imgs0,imgs1,lines,samps = data
        for idx in range(imgs0.shape[0]):
            img0,img1,line,samp = imgs0[idx],imgs1[idx],lines[idx][0].item(),samps[idx][0].item()
            kpts0,kpts1 = match_one_pair(model,img0,img1)
            kpts0 = kpts0[:,[1,0]]
            kpts1 = kpts1[:,[1,0]]
            kpts0[:,0] += line
            kpts0[:,1] += samp
            kpts1[:,0] += line
            kpts1[:,1] += samp
            residuals = np.clip(np.round(np.linalg.norm(kpts0 - kpts1,axis=1)),a_min=0,a_max=255).astype(np.uint8)
            res0[kpts0[:,0].astype(int),kpts0[:,1].astype(int)] = residuals
            res1[kpts1[:,0].astype(int),kpts1[:,1].astype(int)] = residuals
            kpts0_total.append(kpts0)
            kpts1_total.append(kpts1)
            residuals_total.append(residuals)
            pbar.update(1)
            count += 1

            img0 = img0.numpy()
            img1 = img1.numpy()

            for i in range(len(kpts0)):
                cv2.circle(img0,(int(kpts0[i,0]),int(kpts0[i,1])),1,(0,255,0),-1)
                cv2.circle(img1,(int(kpts1[i,0]),int(kpts1[i,1])),1,(0,255,0),-1)
            
            cv2.imwrite(os.path.join(output_path,f'img0_{count}.png'),img0)
            cv2.imwrite(os.path.join(output_path,f'img1_{count}.png'),img1)

            if count % 10 == 0:
                np.save(os.path.join(output_path,'res_1.npy'),res0)
                np.save(os.path.join(output_path,'res_2.npy'),res1)
                exit()


    np.save(os.path.join(output_path,'res_1.npy'),res0)
    np.save(os.path.join(output_path,'res_2.npy'),res1)
    kpts0_total = np.concatenate(kpts0_total,axis=0)
    kpts1_total = np.concatenate(kpts1_total,axis=0)
    residuals_total = np.concatenate(residuals_total,axis=0)

    with open(os.path.join(output_path,'match_res.txt'),'w') as f:
        for kpt0,kpt1,res in zip(kpts0_total,kpts1_total,residuals_total):
            f.write(f"{kpt0[0]:.2f} {kpt0[1]:.2f} {kpt1[0]:.2f} {kpt1[1]:.2f} {res}\n")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

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

    imgs = [os.path.join(options.root,i) for i in os.listdir(options.root) if 'tif' in i]
    if len(imgs) != 2:
        raise ValueError("错误：输入文件夹中需要正好包含两张影像")
    
    match(model,imgs[0],imgs[1],options.root)