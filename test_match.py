import cv2
import torch
import matplotlib.pyplot as plt
import argparse
import os
import rasterio
import rasterio.errors
import numpy as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

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
                img_data = np.stack([img_data]*3,axis=-1)
            else:
                # 读取所有波段数据，并计算平均值进行全色化
                multispectral_data = self._src.read(window=window)
                # 计算所有波段的平均值
                img_data = np.mean(multispectral_data, axis=0).astype(multispectral_data.dtype)
                img_data = np.stack([img_data]*3,axis=-1)
            
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
        
        self.size = 512
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

def match(model,tif_path0,tif_path1,output_path,batch_size=8):
    image0 = Image(tif_path0)
    image1 = Image(tif_path1)
    dataset = ImageDataset(image0,image1)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    patch_num = len(dataset)
    pbar = tqdm(total=patch_num)
    count = 0

    for data in dataloader:
        imgs0,imgs1,lines,samps = data
        for idx in range(imgs0.shape[0]):
            img0,img1,line,samp = imgs0[idx],imgs1[idx],lines[idx][0].item(),samps[idx][0].item()
            num_keypoints = 2048
            print(img0.shape,img1.shape)
            points_tensor = model(img0,img1,num_keypoints)
            kpts0 = points_tensor[:, :2].cpu().numpy()
            kpts1 = points_tensor[:, 2:].cpu().numpy()

            img0 = img0.numpy()
            img1 = img1.numpy()

            for i in range(len(kpts0)):
                cv2.circle(img0,(int(kpts0[i,0]),int(kpts0[i,1])),1,(0,255,0),-1)
                cv2.circle(img1,(int(kpts1[i,0]),int(kpts1[i,1])),1,(0,255,0),-1)
            
            img_output_path = os.path.join(output_path,'vis_imgs')
            os.makedirs(img_output_path,exist_ok=True)
            cv2.imwrite(os.path.join(img_output_path,f'img0_{count}.png'),img0)
            cv2.imwrite(os.path.join(img_output_path,f'img1_{count}.png'),img1)
            pbar.update(1)

            if count > 10:
                exit()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    # Load the TorchScript model
    model = torch.jit.load('./weights/mapglue_model.pt')
    model.eval()
    print("Model loaded successfully!")

    imgs = [os.path.join(options.root,i) for i in os.listdir(options.root) if 'tif' in i]
    if len(imgs) != 2:
        raise ValueError("错误：输入文件夹中需要正好包含两张影像")
    output_path = os.path.join(options.root,'match_res')
    os.makedirs(output_path,exist_ok=True)
    match(model,imgs[0],imgs[1],output_path)

