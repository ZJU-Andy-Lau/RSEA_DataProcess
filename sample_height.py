import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import rasterio.errors
import numpy as np
import os
from glob import glob
import argparse

def generate_dem_with_gravity_anomaly(
    remote_sensing_image_path: str,
    dem_folder_path: str,
    gravity_anomaly_image_path: str,
    output_dem_path: str
) -> None:
    """
    根据遥感影像的参考，整合DEM数据，并在重力异常数据中采样叠加，最后导出新的DEM。

    Args:
        remote_sensing_image_path (str): 输入遥感影像tif的路径。
        dem_folder_path (str): 包含若干DEM影像tif的文件夹路径。
                                文件夹结构示例: dem_folder/xxxxx/xxxxx.tif
        gravity_anomaly_image_path (str): 全球重力异常tif数据的路径。
        output_dem_path (str): 导出最终DEM的路径。
    """
    
    print(f"开始处理: {remote_sensing_image_path}")
    print(f"DEM文件夹: {dem_folder_path}")
    print(f"重力异常数据: {gravity_anomaly_image_path}")
    print(f"输出路径: {output_dem_path}")

    # 1. 读取遥感影像的地理参考信息
    try:
        with rasterio.open(remote_sensing_image_path) as src_rs:
            crs = src_rs.crs
            transform = src_rs.transform
            width = src_rs.width
            height = src_rs.height
            bounds = src_rs.bounds
            print(f"遥感影像信息：CRS={crs}, 宽度={width}, 高度={height}")
    except rasterio.errors.RasterioIOError as e:
        print(f"错误: 无法打开遥感影像文件 {remote_sensing_image_path}。请检查路径或文件是否损坏。")
        raise e

    # 创建一个与遥感影像相同形状的空白DEM数组，用于存储整合后的DEM数据
    # 注意：这里初始化为NaN或一个特定的NoData值，以便后续处理
    target_dem_data = np.full((height, width), np.nan, dtype=np.float32)

    # 2. 整合DEM数据并重采样
    # 查找所有DEM文件
    dem_files = glob(os.path.join(dem_folder_path, '**', '*.tif'), recursive=True)
    if not dem_files:
        print(f"警告: 在文件夹 {dem_folder_path} 中没有找到任何DEM文件。")
        # 可以选择在这里抛出错误或者返回，取决于业务需求
        # raise FileNotFoundError(f"在 {dem_folder_path} 中没有找到任何DEM文件。")

    print(f"找到 {len(dem_files)} 个DEM文件，开始重采样和整合...")

    for i, dem_file in enumerate(dem_files):
        try:
            with rasterio.open(dem_file) as src_dem:
                # 检查DEM的CRS是否与遥感影像的CRS兼容
                # 如果不兼容，rasterio会自动进行投影变换，但提前检查有助于理解问题
                if src_dem.crs != crs:
                    print(f"注意: DEM文件 {dem_file} 的CRS ({src_dem.crs}) 与遥感影像CRS ({crs}) 不一致，将进行投影变换。")
                
                # 读取DEM数据
                dem_data = src_dem.read(1) # 读取第一个波段

                # 计算DEM的地理范围在目标CRS下的像素坐标
                # 这一步是为了确定当前DEM在目标_dem_data中的有效写入区域
                
                # 使用reproject进行重采样和投影
                # destination参数用于指定输出数组，可以直接写入target_dem_data的相应区域
                # 这里我们先重采样到一个临时数组，然后将有效数据合并到target_dem_data中
                reprojected_dem_data = np.empty((height, width), dtype=np.float32)
                reproject(
                    source=dem_data,
                    destination=reprojected_dem_data,
                    src_transform=src_dem.transform,
                    src_crs=src_dem.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear, # 双线性插值，适用于DEM数据
                    num_threads=os.cpu_count() or 1 # 使用所有CPU核心进行加速
                )
                
                # 将重采样后的DEM数据合并到目标DEM数组中
                # 优先保留非NaN值，或者按特定规则合并（例如，取平均值或最后一个值）
                # 这里我们简单地将非NaN值覆盖，如果多个DEM重叠，后面的会覆盖前面的
                # 如果需要更复杂的合并逻辑，可以在这里实现，例如取平均值等
                nan_mask = np.isnan(target_dem_data)
                target_dem_data[nan_mask] = reprojected_dem_data[nan_mask]
                
                # 对于那些已经被填充的区域，可以根据需要决定如何处理重叠
                # 例如，如果重采样后的DEM数据有值，而目标区域已经有值，可以取平均值
                # current_valid = ~np.isnan(reprojected_dem_data)
                # target_valid = ~np.isnan(target_dem_data)
                # overlap = current_valid & target_valid
                # target_dem_data[overlap] = (target_dem_data[overlap] + reprojected_dem_data[overlap]) / 2.0
                # target_dem_data[current_valid & ~target_valid] = reprojected_dem_data[current_valid & ~target_valid]

            print(f"  - 已处理DEM文件 {i+1}/{len(dem_files)}: {os.path.basename(dem_file)}")
        except rasterio.errors.RasterioIOError as e:
            print(f"错误: 无法打开或处理DEM文件 {dem_file}。跳过此文件。错误信息: {e}")
        except Exception as e:
            print(f"处理DEM文件 {dem_file} 时发生未知错误: {e}")

    # 检查是否有NaN值存在，并处理NoData
    if np.any(np.isnan(target_dem_data)):
        print("警告: 整合后的DEM数据中存在NaN值，这可能是由于某些区域没有对应的DEM数据覆盖。")
        # 可以选择用一个特定值填充NaN，例如0或遥感影像的NoData值
        # target_dem_data[np.isnan(target_dem_data)] = 0 # 或者其他合适的值

    # 3. 重力异常数据采样与叠加
    try:
        with rasterio.open(gravity_anomaly_image_path) as src_gravity:
            gravity_anomaly_data = src_gravity.read(1)

            # 获取遥感影像每个像素的中心坐标
            rows, cols = np.indices((height, width))
            x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)
            
            # 将x_coords和y_coords展平，以便进行采样
            x_flat = x_coords.flatten()
            y_flat = y_coords.flatten()
            
            # 使用rasterio的sample函数进行采样
            # 需要将坐标从遥感影像的CRS转换到重力异常数据的CRS
            # 如果CRS不同，reproject_coords 会处理坐标转换
            
            # 获取重力异常数据的transform和crs
            gravity_transform = src_gravity.transform
            gravity_crs = src_gravity.crs

            # 转换坐标到重力异常数据的CRS
            if crs != gravity_crs:
                print(f"注意: 遥感影像CRS ({crs}) 与重力异常CRS ({gravity_crs}) 不一致，将进行坐标转换。")
                # 将遥感影像的每个像素中心坐标从其CRS转换到重力异常的CRS
                # 注意：reproject_coords 返回的是一个生成器，需要列表化
                reprojected_coords = list(rasterio.warp.reproject_coords(
                    x_flat, y_flat, crs, gravity_crs
                ))
                x_gravity_coords = reprojected_coords[0]
                y_gravity_coords = reprojected_coords[1]
            else:
                x_gravity_coords = x_flat
                y_gravity_coords = y_flat

            # 使用sample函数进行采样
            # sample 返回一个生成器，每个元素是一个元组 (value,)
            sampled_gravity_values = np.array([
                val[0] for val in src_gravity.sample(zip(x_gravity_coords, y_gravity_coords))
            ]).reshape(height, width)
            
            # 将采样到的重力异常值叠加到DEM上
            # 确保数据类型兼容
            if target_dem_data.dtype != sampled_gravity_values.dtype:
                target_dem_data = target_dem_data.astype(sampled_gravity_values.dtype)
                
            target_dem_data += sampled_gravity_values

            print("重力异常数据采样并叠加完成。")

    except rasterio.errors.RasterioIOError as e:
        print(f"错误: 无法打开重力异常文件 {gravity_anomaly_image_path}。请检查路径或文件是否损坏。")
        raise e
    except Exception as e:
        print(f"处理重力异常数据时发生未知错误: {e}")
        raise e

    # 4. 导出最终的DEM文件
    try:
        # 定义输出文件的元数据
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'crs': crs,
            'transform': transform,
            'dtype': target_dem_data.dtype,
            'count': 1, # 单波段
            'nodata': np.nan # 可以根据需要设置NoData值，这里与初始化相同
        }

        with rasterio.open(output_dem_path, 'w', **profile) as dst:
            dst.write(target_dem_data, 1)
            print(f"成功导出DEM文件到: {output_dem_path}")
    except Exception as e:
        print(f"导出DEM文件时发生错误: {e}")
        raise e

# --- 使用示例 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    remote_sensing_image = [os.path.join(options.root,i) for i in os.listdir(options.root) if 'overlay_1' in i][0]
    dem_folder = './data/dem_repo'
    gravity_anomaly_image = './data/egm.tif'
    output_dem_file = os.path.join(options.root,'dem.tif')

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_dem_file), exist_ok=True)

    try:
        generate_dem_with_gravity_anomaly(
            remote_sensing_image,
            dem_folder,
            gravity_anomaly_image,
            output_dem_file
        )
        print("\n函数执行完毕。请检查输出文件。")
    except Exception as e:
        print(f"\n函数执行失败: {e}")
