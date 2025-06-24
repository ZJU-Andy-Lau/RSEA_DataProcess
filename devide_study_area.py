import os
import glob
import rasterio
import rasterio.features
import rasterio.errors
import rasterio.warp
import rasterio.transform # 导入 transform 模块
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import transform, unary_union
import pyproj
from pyproj import CRS, Transformer
import warnings

import argparse

# Suppress warnings from shapely, pyproj etc. if they are not critical
warnings.filterwarnings("ignore")

# --- Configuration ---
MIN_OVERLAP_SIDE_METERS = 3000             # 内接矩形最小边长要求，单位：米
OUTPUT_NODATA_VALUE = 0                    # 输出影像的 NoData 值

# --- Helper Functions ---

def get_image_footprint_and_crs(image_path):
    """
    提取影像的有效数据区域（多边形）和坐标参考系统 (CRS)，排除 NoData 区域。
    返回的是在影像原始 CRS 下的足迹多边形。
    """
    try:
        with rasterio.open(image_path) as src:
            nodata_val = src.nodata
            
            # 默认使用影像的边界框作为足迹
            footprint = box(*src.bounds)

            if nodata_val is not None:
                # 读取第一个波段的掩膜。255表示有效数据，0表示NoData。
                mask = src.read_masks(1) 
                
                valid_shapes = [
                    Polygon(geom['coordinates'][0]) for geom, val in rasterio.features.shapes(mask, transform=src.transform) if val == 255
                ]
                
                if not valid_shapes:
                    print(f"警告：影像 {os.path.basename(image_path)} 未找到有效数据形状（可能是全NoData）。将使用影像边界作为足迹。")
                else:
                    try:
                        footprint = unary_union(valid_shapes)
                    except Exception as e:
                        print(f"警告：合并影像 {os.path.basename(image_path)} 的有效数据形状失败。错误：{e}。将使用影像边界作为足迹。")
            
            return footprint, src.crs
    except rasterio.errors.RasterioIOError as e:
        print(f"错误：无法打开影像 {image_path} 或读取其元数据。错误信息：{e}")
        return None, None
    except Exception as e:
        print(f"处理影像 {image_path} 时发生未知错误：{e}")
        return None, None

def reproject_polygon(polygon, src_crs, dst_crs):
    """
    将 Shapely 多边形从源 CRS 重投影到目标 CRS。
    """
    if not src_crs or not dst_crs:
        raise ValueError("源 CRS 或目标 CRS 无效，无法进行重投影。")
    
    try:
        # always_xy=True 确保输出坐标是 (x, y) 顺序，即 (经度, 纬度) 或 (东坐标, 北坐标)
        project_transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return transform(project_transformer.transform, polygon)
    except pyproj.exceptions.CRSError as e:
        print(f"CRS 重投影错误：{e}")
        raise ValueError(f"无法从 {src_crs.to_epsg()} 重投影到 {dst_crs.to_epsg()}")

def get_suitable_utm_crs(bounds_lonlat):
    """
    根据地理边界（经纬度）确定一个合适的 UTM CRS (EPSG 代码)。
    """
    min_lon, min_lat, max_lon, max_lat = bounds_lonlat
    
    center_lon = (min_lon + max_lon) / 2

    if min_lat < 0: # 南半球
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = 32700 + utm_zone # 南半球 UTM EPSG 代码范围
    else: # 北半球
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = 32600 + utm_zone # 北半球 UTM EPSG 代码范围
    
    try:
        utm_crs = CRS.from_epsg(epsg_code)
        return utm_crs
    except Exception:
        print(f"警告：无法为边界 {bounds_lonlat} 找到合适的 UTM CRS (EPSG:{epsg_code})。退回到通用投影 CRS (EPSG:3857)。")
        return CRS.from_epsg(3857) 

def find_largest_inscribed_axis_aligned_rectangle(polygon_proj, target_pixel_size_meters, min_side_meters): # 修正：添加 min_side_meters 参数
    """
    在一个投影坐标系下的多边形中，找到最大的轴对齐内接矩形。
    使用栅格化和最大直方图矩形算法。
    polygon_proj: 在投影 CRS 下的 Shapely Polygon 或 MultiPolygon。
    target_pixel_size_meters: 用于栅格化的目标像素大小（米）。
    min_side_meters: 最小边长要求，用于最终尺寸检查。
    返回一个 Shapely Box 对象 (在投影 CRS 下)，如果未找到则返回 None。
    """
    if polygon_proj.is_empty:
        return None

    # 1. 确定栅格化范围和尺寸
    minx, miny, maxx, maxy = polygon_proj.bounds
    
    # 为了避免浮点数误差导致边界问题，稍微扩大栅格化范围
    buffer_meters = target_pixel_size_meters * 2 # 2个像素的缓冲区
    minx -= buffer_meters
    miny -= buffer_meters
    maxx += buffer_meters
    maxy += buffer_meters

    # 计算栅格化尺寸和变换
    width_pixels = int(np.ceil((maxx - minx) / target_pixel_size_meters))
    height_pixels = int(np.ceil((maxy - miny) / target_pixel_size_meters))
    
    # 确保最小尺寸，避免空数组
    if width_pixels <= 0 or height_pixels <= 0:
        return None

    temp_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width_pixels, height_pixels)

    # 2. 栅格化多边形到掩膜
    # `all_touched=True` 确保与多边形接触的所有像素都被标记，这对于寻找内接矩形很重要
    mask = rasterio.features.rasterize(
        shapes=[(polygon_proj, 255)], # 255 表示有效区域
        out_shape=(height_pixels, width_pixels),
        transform=temp_transform,
        fill=0, # 0 表示 NoData
        all_touched=True,
        dtype='uint8'
    )
    
    # 3. 在掩膜中寻找最大矩形 (Maximum Rectangle in a Binary Matrix)
    # 转换为布尔矩阵，True表示有效数据，False表示NoData
    binary_matrix = (mask == 255)

    max_area = 0
    best_rect_pixels = None # (x_min, y_min, x_max, y_max) 像素坐标

    # 辅助数组，存储当前列每个位置向上的连续True的数量
    heights = np.zeros(width_pixels, dtype=int)

    for r in range(height_pixels):
        for c in range(width_pixels):
            if binary_matrix[r, c]:
                heights[c] += 1
            else:
                heights[c] = 0
        
        # 对于当前行，将 heights 视为一个直方图，找到其中最大的矩形
        stack = [] # 存储 (index, height)
        for c_idx in range(width_pixels + 1): # 遍历到 width_pixels + 1 以处理栈中剩余的元素
            current_h = heights[c_idx] if c_idx < width_pixels else 0
            start_c = c_idx
            while stack and current_h < stack[-1][1]:
                # 弹出比当前高度高的柱子，计算以其为高的矩形面积
                prev_c, prev_h = stack.pop()
                width = c_idx - prev_c
                area = prev_h * width
                if area > max_area:
                    max_area = area
                    # 像素坐标：(左上角x, 左上角y, 右下角x, 右下角y)
                    best_rect_pixels = (prev_c, r - prev_h + 1, c_idx - 1, r)
                start_c = prev_c # 更新当前柱子可以延伸到的最左边

            stack.append((start_c, current_h))

    if best_rect_pixels is None:
        return None

    # 4. 将像素坐标的矩形转换为地理坐标的 Shapely Box
    px_min, py_min, px_max, py_max = best_rect_pixels

    # 根据栅格化时的 transform，将像素坐标转换为地理坐标
    
    # 左上角 (minx, maxy)
    ul_x, ul_y = temp_transform * (px_min, py_min)
    # 右下角 (maxx, miny)
    lr_x, lr_y = temp_transform * (px_max + 1, py_max + 1) # +1是因为像素边界

    # 矫正由于y_res通常为负导致的问题
    # ul_y是maxy，lr_y是miny
    true_minx = ul_x
    true_maxy = ul_y
    true_maxx = lr_x
    true_miny = lr_y

    inscribed_rect_proj = box(true_minx, true_miny, true_maxx, true_maxy)

    # 检查矩形尺寸是否满足要求 (在投影 CRS 下)
    width_rect = inscribed_rect_proj.bounds[2] - inscribed_rect_proj.bounds[0]
    height_rect = inscribed_rect_proj.bounds[3] - inscribed_rect_proj.bounds[1]

    # 修正：在这里使用传入的 min_side_meters 进行检查
    if width_rect >= min_side_meters and height_rect >= min_side_meters: 
        return inscribed_rect_proj
    else:
        return None

def calculate_overlap_and_inscribed_rectangle(poly1_orig_crs, crs1, poly2_orig_crs, crs2, min_side_meters):
    """
    计算两个多边形的重叠区域，找到重叠区域内最大的轴对齐内接矩形，
    并检查其边长是否满足最小长度要求。
    如果符合要求，返回在 poly1 原始 CRS 下的内接矩形；否则返回 None。
    """
    latlon_crs = CRS.from_epsg(4326) # WGS84 地理 CRS

    try:
        poly1_latlon = reproject_polygon(poly1_orig_crs, crs1, latlon_crs)
        poly2_latlon = reproject_polygon(poly2_orig_crs, crs2, latlon_crs)
    except ValueError as e:
        print(f"重投影到经纬度 CRS 失败: {e}. 跳过当前重叠计算。")
        return None

    combined_bounds_latlon = poly1_latlon.union(poly2_latlon).bounds
    
    proj_crs = get_suitable_utm_crs(combined_bounds_latlon)

    try:
        poly1_proj = reproject_polygon(poly1_orig_crs, crs1, proj_crs)
        poly2_proj = reproject_polygon(poly2_orig_crs, crs2, proj_crs)
    except ValueError as e:
        print(f"重投影到 UTM CRS 失败: {e}. 跳过当前重叠计算。")
        return None

    intersection_proj = poly1_proj.intersection(poly2_proj)

    if intersection_proj.is_empty or not (isinstance(intersection_proj, Polygon) or isinstance(intersection_proj, unary_union)): 
        return None 
    
    # 估算一个合适的栅格化像素大小，可以取两张影像分辨率的平均值或一个固定小值
    # 如果原始影像分辨率很高，这个值应该更小。这里我们取一个经验值，例如 10 米。
    pixel_size_estimation = 10 # 假设10米分辨率用于内接矩形栅格化查找

    # 查找最大的轴对齐内接矩形，并将 min_side_meters 传入
    inscribed_rect_proj = find_largest_inscribed_axis_aligned_rectangle(
        intersection_proj, pixel_size_estimation, min_side_meters # 修正：传入 min_side_meters
    )

    if inscribed_rect_proj:
        try:
            # 将内接矩形重投影回 img1 的原始 CRS
            reproject_back_transformer = Transformer.from_crs(proj_crs, crs1, always_xy=True)
            return transform(reproject_back_transformer.transform, inscribed_rect_proj)
        except ValueError as e:
            print(f"重投影内接矩形到原始 CRS 失败: {e}")
            return None
    else:
        return None

def process_overlap(img1_path, img2_path, output_dir, inscribed_rectangle_original_crs):
    """
    处理一对重叠影像：将其裁剪到精确的内接矩形区域，转换为全色（灰度）图像，
    并将像素值拉伸到 0-255 (uint8)，然后保存输出。
    inscribed_rectangle_original_crs 是在 img1 CRS 下的内接矩形。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确定目标输出影像的 CRS、变换和尺寸
    # 我们将以 img1 的 CRS 作为目标输出的 CRS，并以 inscribed_rectangle_original_crs 为其地理边界
    with rasterio.open(img1_path) as src1:
        target_crs = src1.crs
        # 计算目标输出的像素大小，可以取 img1 的平均分辨率，或者根据需求固定一个
        # 这里为了确保精确裁剪后的像素对齐，我们直接用内接矩形的地理尺寸和期望的像素尺寸来计算
        # 可以假设输出分辨率与输入一致
        avg_pixel_size = (abs(src1.res[0]) + abs(src1.res[1])) / 2
        
        # 根据 inscribed_rectangle_original_crs 计算输出的宽度和高度（像素）
        # inscribed_rectangle_original_crs.bounds 是 (minx, miny, maxx, maxy)
        output_width_meters = inscribed_rectangle_original_crs.bounds[2] - inscribed_rectangle_original_crs.bounds[0]
        output_height_meters = inscribed_rectangle_original_crs.bounds[3] - inscribed_rectangle_original_crs.bounds[1]

        target_width_pixels = int(np.ceil(output_width_meters / avg_pixel_size))
        target_height_pixels = int(np.ceil(output_height_meters / avg_pixel_size))

        # 计算目标输出的仿射变换
        # 左上角坐标是 inscribed_rectangle_original_crs 的 minx, maxy (因为y_res是负值，对应的是最大y值)
        target_transform = rasterio.transform.from_bounds(
            inscribed_rectangle_original_crs.bounds[0],  # minx
            inscribed_rectangle_original_crs.bounds[1],  # miny
            inscribed_rectangle_original_crs.bounds[2],  # maxx
            inscribed_rectangle_original_crs.bounds[3],  # maxy
            target_width_pixels, 
            target_height_pixels
        )

    # 循环处理两张影像
    for i, img_path in enumerate([img1_path, img2_path]):
        output_filename = os.path.join(output_dir, f"overlap_{i+1}_{os.path.basename(img_path)}")
        
        try:
            with rasterio.open(img_path) as src:
                # 1. 获取当前影像的有效足迹 (这里其实在主流程中已经获取过，但为了函数独立性，可以再次获取或传入)
                # 实际上，inscribed_rectangle_original_crs 已经是确保无 NoData 的区域，
                # 所以此处无需再额外获取 src_footprint 和做交集，直接reproject即可。
                # 但为了健壮性，我们可以确保即使传入的矩形理论上有问题，也能通过 src_nodata 排除。
                
                # 2. 读取源影像所有波段数据
                src_data = src.read()
                
                # 3. 初始化目标数据数组，类型为原始影像类型
                target_data_original_dtype = np.zeros((src.count, target_height_pixels, target_width_pixels), dtype=src.dtypes[0])
                
                # 使用 reproject 进行重采样和裁剪
                rasterio.warp.reproject(
                    source=src_data,
                    destination=target_data_original_dtype,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata, 
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=OUTPUT_NODATA_VALUE, # 输出影像的 NoData 值
                    resampling=Resampling.nearest, # 最邻近插值，保持原始像素值
                    num_threads=os.cpu_count() or 1
                )

                # 4. 转换为全色（灰度）图像
                if target_data_original_dtype.shape[0] > 1:
                    pan_data_original_dtype = np.mean(target_data_original_dtype, axis=0)
                else:
                    pan_data_original_dtype = target_data_original_dtype[0, :, :] 

                # 5. 将像素值拉伸到 0-255 并转换为 uint8
                # 找到非 NoData 区域的最小值和最大值进行拉伸
                valid_pixels = pan_data_original_dtype[pan_data_original_dtype != OUTPUT_NODATA_VALUE]
                
                if valid_pixels.size == 0: # 如果整个区域都是 NoData
                    stretched_pan_data = np.full(pan_data_original_dtype.shape, OUTPUT_NODATA_VALUE, dtype=np.uint8)
                else:
                    min_val = np.min(valid_pixels)
                    max_val = np.max(valid_pixels)

                    if max_val == min_val: # 避免除以零，如果所有有效像素值都相同
                        stretched_pan_data = np.full(pan_data_original_dtype.shape, 127, dtype=np.uint8) # 设为中间灰度
                        stretched_pan_data[pan_data_original_dtype == OUTPUT_NODATA_VALUE] = OUTPUT_NODATA_VALUE # 重新设置NoData
                    else:
                        stretched_pan_data = (pan_data_original_dtype - min_val) / (max_val - min_val) * 255
                        stretched_pan_data = np.clip(stretched_pan_data, 0, 255).astype(np.uint8)
                        # 将 NoData 区域再次设为 OUTPUT_NODATA_VALUE
                        stretched_pan_data[pan_data_original_dtype == OUTPUT_NODATA_VALUE] = OUTPUT_NODATA_VALUE


                # 更新输出文件的元数据
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": target_height_pixels,
                    "width": target_width_pixels,
                    "transform": target_transform,
                    "crs": target_crs,
                    "count": 1, 
                    "dtype": np.uint8, # 修正：输出为 uint8
                    "nodata": OUTPUT_NODATA_VALUE, 
                    "compress": "LZW", 
                    "bigtiff": "YES" if stretched_pan_data.nbytes > 4 * (1024**3) else "NO" 
                })
                
                with rasterio.open(output_filename, "w", **out_meta) as dest:
                    dest.write(stretched_pan_data, 1) 

                print(f"  裁剪并保存了影像 {os.path.basename(img_path)} 的重叠部分到 {output_filename}")
        except rasterio.errors.RasterioIOError as e:
            print(f"错误：处理影像 {img_path} 时发生 IO 错误。错误信息：{e}")
        except Exception as e:
            print(f"处理影像 {img_path} 的重叠部分时发生未知错误：{e}")


# --- Main Script ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    INPUT_FOLDER = os.path.join(options.root,'raw')
    OUTPUT_FOLDER = os.path.join(options.root,'pairs')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.tif'))
    if not image_paths:
        print(f"在 {INPUT_FOLDER} 中未找到任何 .tif 影像。程序退出。")
        exit()

    print(f"共找到 {len(image_paths)} 张影像。开始处理...")

    image_data = {} 
    for i, img_path in enumerate(image_paths):
        print(f"({i+1}/{len(image_paths)}) 正在提取影像足迹：{os.path.basename(img_path)}")
        footprint, crs = get_image_footprint_and_crs(img_path)
        if footprint is not None and crs is not None:
            image_data[img_path] = (footprint, crs)
        else:
            print(f"未能为影像 {os.path.basename(img_path)} 提取有效的足迹或 CRS。将跳过此影像。")

    processed_pairs = set() 
    overlap_counter = 0

    image_paths_list = list(image_data.keys()) 
    
    for i in range(len(image_paths_list)):
        for j in range(i + 1, len(image_paths_list)):
            img1_path = image_paths_list[i]
            img2_path = image_paths_list[j]

            pair_key = tuple(sorted((img1_path, img2_path)))
            if pair_key in processed_pairs:
                continue 

            footprint1, crs1 = image_data[img1_path]
            footprint2, crs2 = image_data[img2_path]
            
            print(f"\n正在检查以下影像对的重叠情况：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")

            # calculate_overlap_and_inscribed_rectangle 现在会尝试找到真正的最大轴对齐内接矩形
            inscribed_rectangle_original_crs = calculate_overlap_and_inscribed_rectangle(
                footprint1, crs1, footprint2, crs2, MIN_OVERLAP_SIDE_METERS
            )

            if inscribed_rectangle_original_crs:
                overlap_counter += 1
                print(f"✅ 找到符合要求的内接矩形 (第 {overlap_counter} 对) ：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")
                
                output_pair_dir = os.path.join(
                    OUTPUT_FOLDER,
                    f"overlap_pair_{overlap_counter}_{os.path.basename(os.path.splitext(img1_path)[0])}_{os.path.basename(os.path.splitext(img2_path)[0])}"
                )
                process_overlap(img1_path, img2_path, output_pair_dir, inscribed_rectangle_original_crs)
                processed_pairs.add(pair_key) 
            else:
                print(f"❌ 未找到符合要求的内接矩形：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")

    print(f"\n--- 处理完成 ---")
    print(f"总计找到并处理了 {overlap_counter} 对符合要求的影像重叠。")
    print(f"所有输出文件保存在：{OUTPUT_FOLDER}")


