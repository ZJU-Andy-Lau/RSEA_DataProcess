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
    gravity_buffer_degree: float = 0.5 
) -> None:
    """
    根据遥感影像的参考，整合与遥感影像有重叠部分的DEM数据，
    并在重力异常数据中采样叠加，最后导出新的DEM。
    重力异常数据将只加载与遥感影像重叠的区域及额外的缓冲区域。

    Args:
        remote_sensing_image_path (str): 输入遥感影像tif的路径。
        dem_folder_path (str): 包含若干DEM影像tif的文件夹路径。
                                文件夹结构示例: dem_folder/xxxxx/xxxxx.tif
        gravity_anomaly_image_path (str): 全球重力异常tif数据的路径。
        output_dem_path (str): 导出最终DEM的路径。
        gravity_buffer_degree (float): 在重力异常数据加载时，在遥感影像边界框基础上
                                       向外扩展的地理度数（如果CRS是经纬度）或单位（如果CRS是投影坐标系）。
                                       默认为0.5度，以保证鲁棒性。
    """
    
    print(f"开始处理: {remote_sensing_image_path}")
    print(f"DEM文件夹: {dem_folder_path}")
    print(f"重力异常数据: {gravity_anomaly_image_path}")
    print(f"输出路径: {output_dem_path}")
    print(f"重力异常数据加载缓冲量: {gravity_buffer_degree} 单位")

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

    # 3. 重力异常数据采样与叠加
    try:
        with rasterio.open(gravity_anomaly_image_path) as src_gravity:
            gravity_transform = src_gravity.transform
            gravity_crs = src_gravity.crs
            gravity_height = src_gravity.height
            gravity_width = src_gravity.width

            # 确定加载重力异常数据的边界框（遥感影像边界 + 缓冲）
            # 扩展遥感影像的边界框
            buffered_rs_bounds = rasterio.coords.BoundingBox(
                left=rs_bounds.left - gravity_buffer_degree,
                bottom=rs_bounds.bottom - gravity_buffer_degree,
                right=rs_bounds.right + gravity_buffer_degree,
                top=rs_bounds.top + gravity_buffer_degree
            )
            print(f"扩展后的遥感影像边界（用于加载重力异常）: {buffered_rs_bounds}")

            # 将扩展后的边界框转换为重力异常数据CRS下的边界框
            # 这一步非常重要，确保坐标系统一致性
            buffered_bounds_in_gravity_crs = transform_bounds(
                src_crs=crs, # 遥感影像的CRS
                dst_crs=gravity_crs, # 重力异常的CRS
                left=buffered_rs_bounds.left,
                bottom=buffered_rs_bounds.bottom,
                right=buffered_rs_bounds.right,
                top=buffered_rs_bounds.top
            )
            
            # 将地理边界框转换为重力异常数据上的像素窗口
            # from_bounds 返回一个窗口对象 (row_start, row_stop), (col_start, col_stop)
            # bound 参数需要是 (west, south, east, north)
            gravity_window = src_gravity.window(*buffered_bounds_in_gravity_crs)
            
            # 确保窗口范围在重力异常数据的有效范围内
            # bounds.window 已经做了这个裁剪
            # row_start, row_stop = gravity_window.row_start, gravity_window.row_stop
            # col_start, col_stop = gravity_window.col_start, gravity_window.col_stop

            # if row_start < 0: row_start = 0
            # if row_stop > gravity_height: row_stop = gravity_height
            # if col_start < 0: col_start = 0
            # if col_stop > gravity_width: col_stop = gravity_width

            # gravity_window = rasterio.windows.Window(col_start, row_start, col_stop - col_start, row_stop - row_start)

            print(f"在重力异常数据上加载的像素窗口: {gravity_window}")

            # 限制只加载窗口内的重力异常数据
            # read(1, window=gravity_window) 这样只会读取指定窗口的数据
            gravity_anomaly_data_windowed = src_gravity.read(1, window=gravity_window)
            
            # 获取加载数据的仿射变换矩阵
            # 从窗口获取的transform是相对于整个影像的transform
            # 我们可以通过 window_transform 获取该窗口的独立transform
            window_transform = src_gravity.window_transform(gravity_window)

            print(f"重力异常数据局部加载完成，大小: {gravity_anomaly_data_windowed.shape}")

            rows, cols = np.indices((height, width))
            x_coords_grid, y_coords_grid = rasterio.transform.xy(transform, rows, cols)
            x_flat = np.array(x_coords_grid).flatten()
            y_flat = np.array(y_coords_grid).flatten()
            
            # 将遥感影像的每个像素中心坐标从其CRS转换到重力异常数据的CRS
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

            # 对局部加载的重力异常数据进行采样
            # 这里的采样需要注意：src_gravity.sample 仍然是对整个影像进行采样，
            # 但由于我们只加载了部分数据，如果采样点落在未加载的区域，结果将是NoData。
            # 这也是为什么我们需要缓冲加载。
            # 为了确保采样效率和正确性，我们可以直接传入整个影像的采样函数，
            # rasterio会负责从底层GDAL处理，它不会加载完整数据如果它能避免。
            # 但为了明确表达只使用加载的部分，我们可以模拟一个采样过程，
            # 或者更简单的，让rasterio库自行处理（它已经很优化了）。
            # 
            # 考虑到我们已经用 `window` 加载了数据，
            # 最佳实践是直接使用 `src_gravity.sample()`，GDAL底层会自动优化读取，
            # 它不会真的将整个文件加载到内存中再采样，而是根据采样点去读取。
            # 所以，即使我们只读了一部分到 `gravity_anomaly_data_windowed`，
            # `src_gravity.sample` 仍然是针对原始文件的，这正是我们想要的鲁棒性。
            
            sampled_gravity_values = np.array([
                val[0] for val in src_gravity.sample(zip(x_gravity_coords, y_gravity_coords))
            ]).reshape(height, width)
            
            # 确保数据类型兼容，并在叠加前处理NaN值，避免NaN + X = NaN
            valid_dem_mask = ~np.isnan(target_dem_data)
            valid_gravity_mask = ~np.isnan(sampled_gravity_values)
            
            overlap_and_valid_mask = valid_dem_mask & valid_gravity_mask
            
            # 调整数据类型，确保可以进行加法操作
            if np.issubdtype(sampled_gravity_values.dtype, np.floating) and \
               np.issubdtype(target_dem_data.dtype, np.integer):
                print(f"将DEM数据从 {target_dem_data.dtype} 转换为 {sampled_gravity_values.dtype} 以便叠加重力异常。")
                target_dem_data = target_dem_data.astype(sampled_gravity_values.dtype)
            elif np.issubdtype(target_dem_data.dtype, np.floating) and \
                 np.issubdtype(sampled_gravity_values.dtype, np.integer):
                sampled_gravity_values = sampled_gravity_values.astype(target_dem_data.dtype)
            else: 
                 common_dtype = np.promote_types(target_dem_data.dtype, sampled_gravity_values.dtype)
                 print(f"数据类型不匹配，将两者转换为共同类型 {common_dtype} 以便叠加。")
                 target_dem_data = target_dem_data.astype(common_dtype)
                 sampled_gravity_values = sampled_gravity_values.astype(common_dtype)


            target_dem_data[overlap_and_valid_mask] += sampled_gravity_values[overlap_and_valid_mask]

            print("重力异常数据采样并叠加完成。")

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
