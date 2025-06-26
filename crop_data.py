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
    # 将经纬度从度转换为弧度
    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)

    # Web Mercator 球体半径 (WGS84 半长轴)
    R = 6378137.0
    
    # X 坐标
    x = R * lon_rad

    # Y 坐标
    # 确保纬度在合理范围内，避免 tan 或 log 出现无穷大
    # 理论上，Web Mercator 的纬度范围约为 -85.05112878 到 85.05112878 度。
    # 钳制纬度可以防止数值不稳定。
    max_lat_rad = torch.deg2rad(torch.tensor(85.05112878)).cuda()
    min_lat_rad = torch.deg2rad(torch.tensor(-85.05112878)).cuda()
    lat_rad = torch.clamp(lat_rad, min=min_lat_rad, max=max_lat_rad)

    y = R * torch.log(torch.tan(np.pi / 4 + lat_rad / 2))

    # 堆叠 X 和 Y 坐标，形成 (..., 2) 的张量
    web_mercator_coords = torch.stack((x, y), dim=-1).cpu().numpy()
    
    return web_mercator_coords

def crop_data(tif_srcs,dem_src,residuals,tl,br):
    t0 = time.perf_counter()
    H,W = br[0] - tl[0], br[1] - tl[1]
    window = Window(tl[1],tl[0],W,H)
    croped_tifs = [src.read(window = window)[0,:H,:W] for src in tif_srcs]
    croped_dem = dem_src.read(window = window).transpose(1,2,0)[:H,:W]
    croped_residuals = [res[tl[0]:br[0],tl[1]:br[1]] for res in residuals]
    t1 = time.perf_counter()
    print("t1:",t1 - t0)

    row_indices,col_indices = np.meshgrid(
            np.arange(tl[0],br[0]),
            np.arange(tl[1],br[1]),
            indexing='ij'
        )
    local = np.stack([row_indices,col_indices],axis=-1)

    t2 = time.perf_counter()
    print("t2:",t2 - t1)

    src = tif_srcs[0]
    if src.crs != CRS("EPSG:4326"):
        print("输入影像坐标系不是WGS84，需要进行转换")
        exit()
    
    transform_window = src.window_transform(window)

    # coords = np.full((H,W,2),np.nan,dtype=np.float64)

    col_coords_center, row_coords_center = np.meshgrid(
            np.arange(W) + 0.5, 
            np.arange(H) + 0.5, 
            indexing='xy' # 确保 (col, row) 顺序对应 Affine.transform
        )
    
    lons,lats = transform_window * (col_coords_center, row_coords_center)

    # for row_idx in range(H):
    #     for col_idx in range(W):
    #         row_ori = tl[0] + row_idx
    #         col_ori = tl[1] + col_idx

    #         x_center,y_center = src.xy(row_ori + .5, col_ori + .5)

    #         coords[row_idx,col_idx,0] = x_center
    #         coords[row_idx,col_idx,1] = y_center

    t3 = time.perf_counter()
    print("t3:",t3 - t2)

    # coords_flat = coords.reshape(-1,2)
    coords = wgs84_to_web_mercator_cuda(lons,lats).reshape(H,W,2)

    t4 = time.perf_counter()
    print("t4:",t4 - t3)

    # crs_ori = src.crs
    # crs_tgt = CRS("EPSG:4531")

    # if crs_ori != crs_tgt:
    #     transformer = Transformer.from_crs(crs_ori,crs_tgt,always_xy=True)
        
    #     transformed_x,transformed_y = transformer.transform(coords_flat[:,0],coords_flat[:,1])
    #     coords = np.stack([transformed_x,transformed_y],axis=-1).reshape(H,W,2)
    
    obj = np.concatenate([coords,croped_dem],axis=-1)

    return croped_tifs,croped_residuals,local,obj

def main(tif_paths,dem_path,residual_paths,output_folder,crop_size = 3000):
    tif_srcs = [rasterio.open(path) for path in tif_paths]
    dem_src = rasterio.open(dem_path)
    residuals = [np.load(path) for path in residual_paths]
    n = len(tif_paths)
    
    H,W = tif_srcs[0].height,tif_srcs[0].width
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
