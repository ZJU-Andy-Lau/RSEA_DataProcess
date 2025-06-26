import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer 
import cv2
import os
import argparse
from tqdm import tqdm
import torch
import time

def wgs84_to_web_mercator_cuda(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    使用 PyTorch 和 CUDA 将 WGS84 经纬度批量转换为 Web Mercator (EPSG:3857) 投影坐标。
    假设输入经纬度已是度数。

    参数:
    lon (torch.Tensor): 经度张量 (度数)。可以是一维或多维。
    lat (torch.Tensor): 纬度张量 (度数)。与经度张量形状相同。

    返回:
    torch.Tensor: Web Mercator 投影坐标张量，形状为 (..., 2)，其中最后一维是 [X, Y]。
    """
    lon,lat = torch.from_numpy(lon).cuda(),torch.from_numpy(lat).cuda()

    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)

    R = 6378137.0
    
    x = R * lon_rad

    max_lat_rad = torch.deg2rad(torch.tensor(85.05112878)).cuda()
    min_lat_rad = torch.deg2rad(torch.tensor(-85.05112878)).cuda()
    lat_rad = torch.clamp(lat_rad, min=min_lat_rad, max=max_lat_rad)

    y = R * torch.log(torch.tan(np.pi / 4 + lat_rad / 2))

    web_mercator_coords = torch.stack((x, y), dim=-1).cpu().numpy()
    
    return web_mercator_coords

def crop_data(tif_srcs,dem_src,residuals,tl,br):
    H,W = br[0] - tl[0], br[1] - tl[1]
    window = Window(tl[1],tl[0],W,H)
    croped_tifs = [src.read(window = window)[0,:H,:W] for src in tif_srcs]
    croped_dem = dem_src.read(window = window).transpose(1,2,0)[:H,:W]
    croped_residuals = [res[tl[0]:br[0],tl[1]:br[1]] for res in residuals]

    if not ((croped_dem.shape[:2] == croped_tifs[0].shape[:2]) and (croped_dem.shape[:2] == croped_residuals[0].shape[:2])):
        print(f"裁切尺寸出错! tif shape:{croped_tifs[0].shape} \t dem shape:{croped_dem.shape} \t residual shape:{croped_residuals[0].shape}")
        exit()

    row_indices,col_indices = np.meshgrid(
            np.arange(tl[0],br[0]),
            np.arange(tl[1],br[1]),
            indexing='ij'
        )
    local = np.stack([row_indices,col_indices],axis=-1)

    src = tif_srcs[0]
    if src.crs != CRS("EPSG:4326"):
        print("输入影像坐标系不是WGS84，需要进行转换")
        exit()
    
    transform_window = src.window_transform(window)

    col_coords_center, row_coords_center = np.meshgrid(
            np.arange(W) + 0.5, 
            np.arange(H) + 0.5, 
            indexing='xy' # 确保 (col, row) 顺序对应 Affine.transform
        )
    
    lons,lats = transform_window * (col_coords_center, row_coords_center)

    coords = wgs84_to_web_mercator_cuda(lons,lats).reshape(H,W,2)
  
    try:
        obj = np.concatenate([coords,croped_dem],axis=-1)
    except Exception as e:
        print("lons:",lons.shape,"lats:",lats.shape)
        print("coords:",coords.shape,"dem:",croped_dem.shape)

    return croped_tifs,croped_residuals,local,obj

def main(tif_paths,dem_path,residual_paths,output_folder,crop_size = 3000):
    tif_srcs = [rasterio.open(path) for path in tif_paths]
    dem_src = rasterio.open(dem_path)
    residuals = [np.load(path) for path in residual_paths]
    n = len(tif_paths)
    
    H,W = tif_srcs[0].height,tif_srcs[0].width
    print(f"已读入 {n} 张影像，H = {H} \t W = {W}")
    init_step = crop_size // 2
    line_step = (H - crop_size) // ((H - crop_size) // init_step)
    samp_step = (W - crop_size) // ((W - crop_size) // init_step)
    lines = np.arange(0,H,line_step)
    samps = np.arange(0,W,samp_step)
    pbar = tqdm(total=len(lines) * len(samps))

    for line in lines:
        for samp in samps:
            croped_tifs,croped_residuals,local,obj = crop_data(tif_srcs,dem_src,residuals,[line,samp],[line + crop_size,samp + crop_size])
            output_path = os.path.join(output_folder,f"{line}_{samp}")
            os.makedirs(output_path,exist_ok=True)
            np.save(os.path.join(output_path,'local.npy'),local)
            np.save(os.path.join(output_path,'obj.npy'),obj)
            for i in range(n):
                cv2.imwrite(os.path.join(output_path,f'iamge_{i}.png'),croped_tifs[i])
                np.save(os.path.join(output_path,f'residual_{i}.npy'),croped_residuals[i])
            pbar.update(1)
            pbar.set_postfix({"line":line,"samp":samp})




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    file_list = os.listdir(options.root)
    tif_paths = []
    residual_paths = []
    tif_paths.append([os.path.join(options.root,i) for i in file_list if "overlap_1" in i][0])
    tif_paths.append([os.path.join(options.root,i) for i in file_list if "overlap_2" in i][0])
    residual_paths.append([os.path.join(options.root,i) for i in file_list if "res_1" in i][0])
    residual_paths.append([os.path.join(options.root,i) for i in file_list if "res_2" in i][0])
    dem_path = os.path.join(options.root,'dem.tif')
    output_folder = os.path.join(options.root,"crop")
    os.makedirs(output_folder,exist_ok=True)
    main(tif_paths,dem_path,residual_paths,output_folder)
