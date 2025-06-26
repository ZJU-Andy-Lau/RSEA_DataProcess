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
    根据遥感影像的参考，整合与遥感影像有重叠部分的DEM数据，
    并在重力异常数据中采样叠加，最后导出新的DEM。

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
            bounds = src_rs.bounds  # 获取遥感影像的边界框
            print(f"遥感影像信息：CRS={crs}, 宽度={width}, 高度={height}, 边界={bounds}")
    except rasterio.errors.RasterioIOError as e:
        print(f"错误: 无法打开遥感影像文件 {remote_sensing_image_path}。请检查路径或文件是否损坏。")
        raise e

    # 创建一个与遥感影像相同形状的空白DEM数组，用于存储整合后的DEM数据
    target_dem_data = np.full((height, width), np.nan, dtype=np.float32)

    # 2. 整合DEM数据并重采样
    dem_files = glob(os.path.join(dem_folder_path, '**', '*.tif'), recursive=True)
    if not dem_files:
        print(f"警告: 在文件夹 {dem_folder_path} 中没有找到任何DEM文件。")
        return

    print(f"找到 {len(dem_files)} 个DEM文件，开始筛选并重采样整合...")

    processed_dem_count = 0
    for i, dem_file in enumerate(dem_files):
        try:
            with rasterio.open(dem_file) as src_dem:
                dem_bounds = src_dem.bounds # 获取当前DEM的边界框

                # **关键优化：判断DEM与遥感影像的边界框是否有重叠**
                # 注意：这里需要确保两个bounds在同一个CRS下进行比较。
                # 由于我们后续会reproject，这里假设可以进行粗略比较，
                # 或者更严谨的做法是将其中一个bounds转换到另一个CRS，但通常这会增加复杂性。
                # 简单起见，我们先直接比较。
                
                # rasterio的Bounds对象有一个intersects方法，可以直接判断是否相交
                # 或者手动判断：
                # dem_left, dem_bottom, dem_right, dem_top = dem_bounds
                # rs_left, rs_bottom, rs_right, rs_top = bounds
                # has_overlap = not (dem_right < rs_left or dem_left > rs_right or
                #                    dem_top < rs_bottom or dem_bottom > rs_top)

                # 使用rasterio.intersects，它更健壮且考虑了CRS差异（如果指定了）
                # 但为了准确性，通常需要先将DEM的bounds投影到遥感影像的CRS
                # 这里我们假设DEM和遥感影像的CRS是兼容的（或者大部分重叠判断在这种情况下是有效的）
                # 更严谨的做法是：
                # from rasterio.warp import transform_bounds
                # dem_bounds_in_rs_crs = transform_bounds(src_dem.crs, crs,
                #                                          dem_bounds.left, dem_bounds.bottom,
                #                                          dem_bounds.right, dem_bounds.top)
                # if not bounds.intersects(rasterio.coords.BoundingBox(*dem_bounds_in_rs_crs)):
                #     print(f"  - DEM文件 {os.path.basename(dem_file)} 没有与遥感影像重叠，跳过。")
                #     continue
                
                # 简化处理：直接使用rasterio.bounds的intersects方法，它在内部会处理不同CRS的情况
                # 但需要注意，如果CRS差异很大，直接比较可能会有误差，建议使用transform_bounds更精确
                
                if not bounds.intersects(dem_bounds):
                    print(f"  - DEM文件 {os.path.basename(dem_file)} ({dem_bounds}) 没有与遥感影像 ({bounds}) 重叠，跳过。")
                    continue

                print(f"  - DEM文件 {i+1}/{len(dem_files)}: {os.path.basename(dem_file)} 与遥感影像有重叠，开始处理。")
                processed_dem_count += 1
                
                # 检查DEM的CRS是否与遥感影像的CRS兼容
                if src_dem.crs != crs:
                    print(f"    注意: DEM文件 {dem_file} 的CRS ({src_dem.crs}) 与遥感影像CRS ({crs}) 不一致，将进行投影变换。")
                
                dem_data = src_dem.read(1)

                reprojected_dem_data = np.empty((height, width), dtype=np.float32)
                reproject(
                    source=dem_data,
                    destination=reprojected_dem_data,
                    src_transform=src_dem.transform,
                    src_crs=src_dem.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    num_threads=os.cpu_count() or 1
                )
                
                # 将重采样后的DEM数据合并到目标DEM数组中
                nan_mask = np.isnan(target_dem_data)
                # 只有当目标位置是NaN且重采样后的DEM有值时才填充
                valid_reprojected = ~np.isnan(reprojected_dem_data)
                target_dem_data[nan_mask & valid_reprojected] = reprojected_dem_data[nan_mask & valid_reprojected]
                
                # 如果需要处理重叠区域的合并逻辑（例如平均值），可以在这里添加
                # 例如：
                # overlap_mask = (~nan_mask) & valid_reprojected
                # target_dem_data[overlap_mask] = (target_dem_data[overlap_mask] + reprojected_dem_data[overlap_mask]) / 2.0


        except rasterio.errors.RasterioIOError as e:
            print(f"错误: 无法打开或处理DEM文件 {dem_file}。跳过此文件。错误信息: {e}")
        except Exception as e:
            print(f"处理DEM文件 {dem_file} 时发生未知错误: {e}")

    print(f"总共处理了 {processed_dem_count} 个与遥感影像有重叠的DEM文件。")
    if processed_dem_count == 0:
        print("警告: 没有找到任何与遥感影像重叠的DEM数据。输出的DEM可能全是NoData。")

    if np.any(np.isnan(target_dem_data)):
        print("警告: 整合后的DEM数据中存在NaN值，这可能是由于某些区域没有对应的DEM数据覆盖。")

    # 3. 重力异常数据采样与叠加 (与之前代码相同)
    try:
        with rasterio.open(gravity_anomaly_image_path) as src_gravity:
            gravity_anomaly_data = src_gravity.read(1)

            rows, cols = np.indices((height, width))
            x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)
            
            x_flat = x_coords.flatten()
            y_flat = y_coords.flatten()
            
            gravity_transform = src_gravity.transform
            gravity_crs = src_gravity.crs

            if crs != gravity_crs:
                print(f"注意: 遥感影像CRS ({crs}) 与重力异常CRS ({gravity_crs}) 不一致，将进行坐标转换。")
                reprojected_coords = list(rasterio.warp.reproject_coords(
                    x_flat, y_flat, crs, gravity_crs
                ))
                x_gravity_coords = reprojected_coords[0]
                y_gravity_coords = reprojected_coords[1]
            else:
                x_gravity_coords = x_flat
                y_gravity_coords = y_flat

            sampled_gravity_values = np.array([
                val[0] for val in src_gravity.sample(zip(x_gravity_coords, y_gravity_coords))
            ]).reshape(height, width)
            
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

    # 4. 导出最终的DEM文件 (与之前代码相同)
    try:
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'crs': crs,
            'transform': transform,
            'dtype': target_dem_data.dtype,
            'count': 1,
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

    remote_sensing_image = [os.path.join(options.root,i) for i in os.listdir(options.root) if 'overlap_1' in i][0]
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
