import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds 
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
    output_dem_path: str,
    # 新增参数：重力异常数据加载的扩展量（地理单位，例如度或米）
    # 在reproject场景下，这个缓冲量可以帮助reproject函数更好地处理边缘效应
    gravity_buffer_degree: float = 0.5 
) -> None:
    """
    根据遥感影像的参考，整合与遥感影像有重叠部分的DEM数据，
    并在重力异常数据中采样叠加，最后导出新的DEM。
    重力异常数据将通过重投影与DEM对齐并叠加。

    Args:
        remote_sensing_image_path (str): 输入遥感影像tif的路径。
        dem_folder_path (str): 包含若干DEM影像tif的文件夹路径。
                                文件夹结构示例: dem_folder/xxxxx/xxxxx.tif
        gravity_anomaly_image_path (str): 全球重力异常tif数据的路径。
        output_dem_path (str): 导出最终DEM的路径。
        gravity_buffer_degree (float): 在重力异常数据重投影时，在遥感影像边界框基础上
                                       向外扩展的地理度数（如果CRS是经纬度）或单位（如果CRS是投影坐标系）。
                                       默认为0.5度，以保证鲁棒性。
    """
    
    print(f"开始处理: {remote_sensing_image_path}")
    print(f"DEM文件夹: {dem_folder_path}")
    print(f"重力异常数据: {gravity_anomaly_image_path}")
    print(f"输出路径: {output_dem_path}")
    print(f"重力异常数据重投影缓冲量: {gravity_buffer_degree} 单位")

    # 1. 读取遥感影像的地理参考信息
    try:
        with rasterio.open(remote_sensing_image_path) as src_rs:
            crs = src_rs.crs
            transform = src_rs.transform
            width = src_rs.width
            height = src_rs.height
            rs_bounds = src_rs.bounds  # 获取遥感影像的边界框
            print(f"遥感影像信息：CRS={crs}, 宽度={width}, 高度={height}, 边界={rs_bounds}")
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
    
    # 将遥感影像的边界框解包，方便后续比较
    rs_left, rs_bottom, rs_right, rs_top = rs_bounds.left, rs_bounds.bottom, rs_bounds.right, rs_bounds.top

    for i, dem_file in enumerate(dem_files):
        try:
            with rasterio.open(dem_file) as src_dem:
                dem_bounds = src_dem.bounds # 获取当前DEM的边界框
                dem_left, dem_bottom, dem_right, dem_top = dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top

                # 判断两个矩形是否不重叠的条件
                no_overlap = (dem_right <= rs_left or  # DEM在RS左边
                              dem_left >= rs_right or  # DEM在RS右边
                              dem_top <= rs_bottom or  # DEM在RS下面 (地理坐标系中，y值越小越靠南)
                              dem_bottom >= rs_top)    # DEM在RS上面

                if no_overlap:
                    print(f"  - DEM文件 {os.path.basename(dem_file)} ({dem_bounds}) 没有与遥感影像 ({rs_bounds}) 重叠，跳过。")
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
                valid_reprojected = ~np.isnan(reprojected_dem_data)
                target_dem_data[nan_mask & valid_reprojected] = reprojected_dem_data[nan_mask & valid_reprojected]
                
        except rasterio.errors.RasterioIOError as e:
            print(f"错误: 无法打开或处理DEM文件 {dem_file}。跳过此文件。错误信息: {e}")
        except Exception as e:
            print(f"处理DEM文件 {dem_file} 时发生未知错误: {e}")

    print(f"总共处理了 {processed_dem_count} 个与遥感影像有重叠的DEM文件。")
    if processed_dem_count == 0:
        print("警告: 没有找到任何与遥感影像重叠的DEM数据。输出的DEM可能全是NoData。")

    if np.any(np.isnan(target_dem_data)):
        print("警告: 整合后的DEM数据中存在NaN值，这可能是由于某些区域没有对应的DEM数据覆盖。")

    # 3. 重力异常数据重投影与叠加
    try:
        with rasterio.open(gravity_anomaly_image_path) as src_gravity:
            # 扩展遥感影像的边界框，作为重投影的源边界
            # 这允许reproject函数从重力异常数据中读取一个稍大的区域，以确保插值边缘的准确性
            buffered_rs_bounds = rasterio.coords.BoundingBox(
                left=rs_bounds.left - gravity_buffer_degree,
                bottom=rs_bounds.bottom - gravity_buffer_degree,
                right=rs_bounds.right + gravity_buffer_degree,
                top=rs_bounds.top + gravity_buffer_degree
            )
            print(f"扩展后的遥感影像边界（用于重投影重力异常）: {buffered_rs_bounds}")

            # 初始化一个与目标DEM相同大小的数组，用于存储重投影后的重力异常数据
            reprojected_gravity_data = np.empty((height, width), dtype=np.float32)

            print("开始重投影重力异常数据...")
            reproject(
                source=rasterio.band(src_gravity, 1), # 指定源数据（波段1）
                destination=reprojected_gravity_data, # 目标数组
                src_transform=src_gravity.transform,
                src_crs=src_gravity.crs,
                dst_transform=transform, # 目标DEM的transform
                dst_crs=crs,             # 目标DEM的crs
                resampling=Resampling.bilinear, # 双线性插值，通常用于连续数据
                num_threads=os.cpu_count() or 1,
                src_nodata=src_gravity.nodata, # 源数据的NoData值
                dst_nodata=np.nan # 目标数据的NoData值
                # 可以通过src_bounds参数限制reproject读取的源数据范围，
                # 但reproject本身在处理大文件时已经很智能，通常不需要手动指定
                # src_bounds=buffered_bounds_in_gravity_crs # 如果需要更严格的限制源读取范围
            )
            print("重力异常数据重投影完成。")
            
            # 确保数据类型兼容，并在叠加前处理NaN值，避免NaN + X = NaN
            valid_dem_mask = ~np.isnan(target_dem_data)
            valid_gravity_mask = ~np.isnan(reprojected_gravity_data)
            
            overlap_and_valid_mask = valid_dem_mask & valid_gravity_mask
            
            # 调整数据类型，确保可以进行加法操作
            if np.issubdtype(reprojected_gravity_data.dtype, np.floating) and \
               np.issubdtype(target_dem_data.dtype, np.integer):
                print(f"将DEM数据从 {target_dem_data.dtype} 转换为 {reprojected_gravity_data.dtype} 以便叠加重力异常。")
                target_dem_data = target_dem_data.astype(reprojected_gravity_data.dtype)
            elif np.issubdtype(target_dem_data.dtype, np.floating) and \
                 np.issubdtype(reprojected_gravity_data.dtype, np.integer):
                reprojected_gravity_data = reprojected_gravity_data.astype(target_dem_data.dtype)
            else: 
                 common_dtype = np.promote_types(target_dem_data.dtype, reprojected_gravity_data.dtype)
                 print(f"数据类型不匹配，将两者转换为共同类型 {common_dtype} 以便叠加。")
                 target_dem_data = target_dem_data.astype(common_dtype)
                 reprojected_gravity_data = reprojected_gravity_data.astype(common_dtype)

            # 执行叠加
            target_dem_data[overlap_and_valid_mask] += reprojected_gravity_data[overlap_and_valid_mask]

            print("重力异常数据叠加完成。")

    except rasterio.errors.RasterioIOError as e:
        print(f"错误: 无法打开重力异常文件 {gravity_anomaly_image_path}。请检查路径或文件是否损坏。")
        raise e
    except Exception as e:
        print(f"处理重力异常数据时发生未知错误: {e}")
        raise e

    # 4. 导出最终的DEM文件
    try:
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'crs': crs,
            'transform': transform,
            'dtype': target_dem_data.dtype, 
            'count': 1,
            'nodata': np.nan 
        }
        if not np.issubdtype(profile['dtype'], np.floating):
            print(f"警告: 输出数据类型 {profile['dtype']} 不支持NaN作为NoData，请考虑指定一个整数NoData值。")
            profile['nodata'] = -9999 

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
