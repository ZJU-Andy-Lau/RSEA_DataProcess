import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer 
import cv2
import os
import argparse

def crop_data(tif_srcs,dem_src,residuals,tl,br):
    H,W = br[0] - tl[0], br[1] - tl[1]
    window = Window(tl[1],tl[0],W,H)
    croped_tifs = [src.read(window = window)[0,:H,:W] for src in tif_srcs]
    croped_dem = dem_src.read(window = window).transpose(1,2,0)[:H,:W]
    croped_residuals = [res[tl[0]:br[0],tl[1]:br[1]] for res in residuals]

    row_indices,col_indices = np.meshgrid(
            np.arange(tl[0],br[0]),
            np.arange(tl[1],br[1]),
            indexing='ij'
        )
    local = np.stack([row_indices,col_indices],axis=-1)

    src = tif_srcs[0]
    coords = np.full((H,W,2),np.nan,dtype=np.float64)

    for row_idx in range(H):
        for col_idx in range(W):
            row_ori = tl[0] + row_idx
            col_ori = tl[1] + col_idx

            x_center,y_center = src.xy(row_ori + .5, col_ori + .5)

            coords[row_ori,col_ori,0] = x_center
            coords[row_ori,col_ori,1] = y_center

    crs_ori = src.crs
    crs_tgt = CRS("EPSG:4531")

    if crs_ori != crs_tgt:
        transformer = Transformer.from_crs(crs_ori,crs_tgt,always_xy=True)
        coords_flat = coords.reshape(-1,2)
        transformed_x,transformed_y = transformer.transform(coords_flat[:,0],coords_flat[:,1])
        coords = np.stack([transformed_x,transformed_y],axis=-1).reshape(H,W,2)
    
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

    for line in range(0,H,line_step):
        for samp in range(0,W,samp_step):
            croped_tifs,croped_residuals,local,obj = crop_data(tif_srcs,dem_src,residuals,[line,samp],[line + crop_size,samp + crop_size])
            output_path = os.path.join(output_folder,f"{line}_{samp}")
            os.makedirs(output_path,exist_ok=True)
            np.save(os.path.join(output_path,'local.npy'),local)
            np.save(os.path.join(output_path,'obj.npy'),obj)
            for i in range(n):
                cv2.imwrite(os.path.join(output_path,f'iamge_{i}.png'),croped_tifs[i])
                np.save(os.path.join(output_path,f'residual_{i}.npy'),croped_residuals[i])




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
