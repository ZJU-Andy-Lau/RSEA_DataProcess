import rasterio
import numpy as np
import argparse
import os
from rasterio.transform import Affine

def downsample_npy_and_save_geotiff(input_tif_path, input_npy_path, output_tif_path, downsample_factor=16):
    """
    Reads a NumPy array, downsamples it by a specified factor using nanmean,
    and saves it as a GeoTIFF with adjusted georeferencing to maintain
    spatial alignment with a reference GeoTIFF.

    Args:
        input_tif_path (str): Path to the reference GeoTIFF file
                              (to get CRS and original transform).
        input_npy_path (str): Path to the input NumPy array file (.npy).
        output_tif_path (str): Path where the output GeoTIFF file will be saved.
        downsample_factor (int): The factor by which to downsample (e.g., 16 for 16x16).
    """
    if downsample_factor <= 0:
        raise ValueError("Downsample factor must be a positive integer.")

    try:
        # 1. 读取参考 TIFF 的几何信息
        with rasterio.open(input_tif_path) as src:
            original_profile = src.profile
            original_transform = src.transform
            original_height = src.height
            original_width = src.width
            original_crs = src.crs

        # 2. 加载 NumPy 数组
        npy_data = np.load(input_npy_path)

        # 确保 NumPy 数组维度与参考 TIFF 的原始维度匹配
        if npy_data.shape != (original_height, original_width):
            raise ValueError(
                f"NumPy array shape {npy_data.shape} does not match "
                f"reference TIFF dimensions ({original_height}, {original_width})."
            )

        # 3. 将数组中为 0 的部分变为 NaN
        # 使用 np.where 更高效地进行条件替换
        npy_data_nan = np.where(npy_data == 0, np.nan, npy_data)

        # 4. 根据下采样因子裁剪数组，丢弃不能整除的部分
        new_height = (original_height // downsample_factor) * downsample_factor
        new_width = (original_width // downsample_factor) * downsample_factor
        cropped_data = npy_data_nan[:new_height, :new_width]

        # 计算下采样后的新维度
        downsampled_height = new_height // downsample_factor
        downsampled_width = new_width // downsample_factor

        # 5. 进行 16x16 窗口的步长为 16 的滑动平均 (nanmean)
        # 重塑数组以进行块操作，然后计算每个块的 nanmean
        # 例如，对于 (H, W) -> (H/16, 16, W/16, 16)
        reshaped_data = cropped_data.reshape(
            downsampled_height, downsample_factor,
            downsampled_width, downsample_factor
        )
        # 轴 1 和 3 是每个 16x16 块的内部维度
        downsampled_data = np.nanmean(reshaped_data, axis=(1, 3))

        # 6. 将下采样后的 NaN 变为 0
        final_output_data = np.nan_to_num(downsampled_data, nan=0.0)

        # 7. 计算新的地理变换 (transform)
        # 像元大小会放大 downsample_factor 倍
        # 仿射变换的参数： (c, a, b, f, d, e) -> x_geo = a*col + b*row + c, y_geo = d*col + e*row + f
        # 对于北向上图像： (x_min, pixel_width, 0, y_max, 0, -pixel_height)
        # 新的像元宽度和高度都将是原来的 downsample_factor 倍
        new_pixel_width = original_transform.a * downsample_factor
        new_pixel_height = original_transform.e * downsample_factor # 注意 e 通常是负值，表示从上到下y坐标减小

        # 图像左上角坐标保持不变 (c 和 f)
        new_transform = Affine(
            new_pixel_width, original_transform.b, original_transform.c,
            original_transform.d, new_pixel_height, original_transform.f
        )

        # 8. 更新 GeoTIFF 配置文件并保存
        new_profile = original_profile.copy()
        new_profile.update(
            height=downsampled_height,
            width=downsampled_width,
            transform=new_transform,
            dtype=final_output_data.dtype,
            count=1,
            nodata=0 # 将 0 设为 nodata 值
        )

        with rasterio.open(output_tif_path, 'w', **new_profile) as dst:
            dst.write(final_output_data, 1)

        print(f"成功将 '{input_npy_path}' 下采样并保存到 '{output_tif_path}'。")

    except FileNotFoundError as e:
        print(f"错误: 找不到指定文件。 {e}")
    except ValueError as e:
        print(f"数据错误: {e}")
    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio I/O 错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()
    file_list = os.listdir(options.root)
    print(file_list)
    npy1_path = [os.path.join(options.root,i) for i in file_list if 'res_1' in i][0]
    npy2_path = [os.path.join(options.root,i) for i in file_list if 'res_2' in i][0]
    tif1_path = [os.path.join(options.root,i) for i in file_list if 'overlap_1' in i][0]
    tif2_path = [os.path.join(options.root,i) for i in file_list if 'overlap_2' in i][0]
    output_path = os.path.join(options.root,'res_vis')
    os.makedirs(output_path,exist_ok=True)
    downsample_npy_and_save_geotiff(tif1_path,npy1_path,os.path.join(output_path,'res_vis_1.tif'))
    downsample_npy_and_save_geotiff(tif2_path,npy2_path,os.path.join(output_path,'res_vis_2.tif'))