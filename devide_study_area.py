import os
import glob
import rasterio
from rasterio.windows import Window
import rasterio.features
import rasterio.errors
from rasterio.enums import Resampling # 虽然在此代码中未使用，但保留以备将来扩展
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import transform, unary_union # 导入 unary_union
import pyproj
from pyproj import CRS, Transformer
import warnings
import argparse

# Suppress warnings from shapely, pyproj etc. if they are not critical
warnings.filterwarnings("ignore")



# --- Helper Functions ---

def get_image_footprint_and_crs(image_path):
    """
    提取影像的有效数据区域（多边形）和坐标参考系统 (CRS)，排除 NoData 区域。
    返回的是在影像原始 CRS 下的足迹多边形。
    """
    try:
        with rasterio.open(image_path) as src:
            # 获取 NoData 值，如果已指定
            nodata_val = src.nodata
            
            # 默认使用影像的边界框作为足迹
            footprint = box(*src.bounds)

            if nodata_val is not None:
                # 读取第一个波段的掩膜。
                # 对于非常大的影像，read_masks() 可能仍然是内存密集型的。
                # 如果遇到内存问题，可能需要考虑分块处理掩膜或简化 NoData 处理。
                mask = src.read_masks(1) 

                # 提取有效数据（非 NoData）的形状
                # 255 表示有效数据，0 表示 NoData
                valid_shapes = [
                    Polygon(geom['coordinates'][0]) for geom, val in rasterio.features.shapes(mask, transform=src.transform) if val == 255
                ]
                
                if not valid_shapes:
                    print(f"警告：影像 {os.path.basename(image_path)} 未找到有效数据形状。将使用影像边界作为足迹。")
                    # 此时 footprint 仍为 src.bounds 对应的 box
                else:
                    try:
                        # 尝试将所有有效数据多边形合并成一个单一的多边形或多重多边形
                        footprint = unary_union(valid_shapes)
                    except Exception as e:
                        print(f"警告：合并影像 {os.path.basename(image_path)} 的有效数据形状失败。错误：{e}。将使用影像边界作为足迹。")
                        # 此时 footprint 仍为 src.bounds 对应的 box
            
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
    # 确保源 CRS 和目标 CRS 是有效的
    if not src_crs or not dst_crs:
        raise ValueError("源 CRS 或目标 CRS 无效，无法进行重投影。")
    
    # from_crs 可能会失败，需要捕获
    try:
        project_transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return transform(project_transformer.transform, polygon)
    except pyproj.exceptions.CRSError as e:
        print(f"CRS 重投影错误：{e}")
        raise ValueError(f"无法从 {src_crs.to_epsg()} 重投影到 {dst_crs.to_epsg()}")

def get_suitable_utm_crs(bounds_lonlat):
    """
    根据地理边界（经纬度）确定一个合适的 UTM CRS (EPSG 代码)。
    这对于以米为单位的精确距离计算至关重要。
    bounds_lonlat 是 (min_lon, min_lat, max_lon, max_lat)。
    """
    min_lon, min_lat, max_lon, max_lat = bounds_lonlat
    
    # 计算近似的中心经度以确定 UTM 区域
    center_lon = (min_lon + max_lon) / 2

    # 根据纬度判断南半球或北半球
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
        # EPSG:3857 (Web Mercator) 是一个常见的全球投影，但在极地地区变形较大，且不是等面积投影，
        # 不适合精确距离计算，但作为备用方案。
        return CRS.from_epsg(3857) 

def calculate_overlap_and_inscribed_rectangle(poly1_orig_crs, crs1, poly2_orig_crs, crs2, min_side_meters):
    """
    计算两个多边形的重叠区域，找到重叠区域的轴对齐边界框，
    并检查其边长是否满足最小长度要求。
    如果符合要求，返回在 poly1 原始 CRS 下的轴对齐重叠矩形；否则返回 None。
    """
    # 1. 确定用于距离计算的通用投影 CRS
    
    # 将 poly1 和 poly2 重投影到经纬度 CRS (WGS84) 以确定准确的 UTM 区域
    latlon_crs = CRS.from_epsg(4326) # WGS84 地理 CRS

    try:
        poly1_latlon = reproject_polygon(poly1_orig_crs, crs1, latlon_crs)
        poly2_latlon = reproject_polygon(poly2_orig_crs, crs2, latlon_crs)
    except ValueError as e:
        print(f"重投影到经纬度 CRS 失败: {e}. 跳过当前重叠计算。")
        return None

    # 获取重叠区域的组合边界（经纬度），以确定合适的 UTM 区域
    combined_bounds_latlon = poly1_latlon.union(poly2_latlon).bounds
    
    # 获取适合精确距离计算的 UTM CRS
    proj_crs = get_suitable_utm_crs(combined_bounds_latlon)

    # 将多边形重投影到这个公共投影 CRS
    try:
        poly1_proj = reproject_polygon(poly1_orig_crs, crs1, proj_crs)
        poly2_proj = reproject_polygon(poly2_orig_crs, crs2, proj_crs)
    except ValueError as e:
        print(f"重投影到 UTM CRS 失败: {e}. 跳过当前重叠计算。")
        return None

    # 2. 计算重叠区域
    intersection_proj = poly1_proj.intersection(poly2_proj)

    # 检查交集是否有效（非空且为多边形类型）
    if intersection_proj.is_empty or not (isinstance(intersection_proj, Polygon) or isinstance(intersection_proj, unary_union)): # unary_union 结果可能是 MultiPolygon
        return None # 无效的或空的多边形重叠

    # 3. 找到重叠区域的轴对齐边界框作为“内接矩形”
    # 这是最简单且保证在交集内部的矩形，用于检查尺寸。
    # 严格的“最大内接矩形”是一个更复杂的几何问题，当前代码采用近似方案。
    minx_int, miny_int, maxx_int, maxy_int = intersection_proj.bounds
    inscribed_rect_proj = box(minx_int, miny_int, maxx_int, maxy_int) # 修正：这里是正确的变量名

    width_proj = inscribed_rect_proj.bounds[2] - inscribed_rect_proj.bounds[0]
    height_proj = inscribed_rect_proj.bounds[3] - inscribed_rect_proj.bounds[1] # 修正：这里也是正确的变量名
    
    if width_proj >= min_side_meters and height_proj >= min_side_meters:
        # 将内接矩形重投影回 poly1 的原始 CRS
        # 这对于后续使用 rasterio.window_from_bounds 裁剪非常重要
        try:
            reproject_back_transformer = Transformer.from_crs(proj_crs, crs1, always_xy=True)
            inscribed_rect_original_crs = transform(reproject_back_transformer.transform, inscribed_rect_proj)
            return inscribed_rect_original_crs
        except ValueError as e:
            print(f"重投影内接矩形到原始 CRS 失败: {e}")
            return None
    else:
        return None

def process_overlap(img1_path, img2_path, output_dir, overlap_bbox_original_crs):
    """
    处理一对重叠影像：将其裁剪到重叠区域的边界框，转换为全色（灰度）图像，并保存输出。
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate([img1_path, img2_path]):
        try:
            with rasterio.open(img_path) as src:
                # 根据 overlap_bbox_original_crs 确定要读取的窗口
                # rasterio.windows.from_bounds 会将地理边界转换为像素坐标的窗口
                window = src.window(*overlap_bbox_original_crs.bounds)

                # 确保窗口有效且在影像边界内
                window = window.intersection(Window(0, 0, src.width, src.height))
                if window.width <= 0 or window.height <= 0:
                    print(f"警告：为影像 {os.path.basename(img_path)} 计算的裁剪窗口为空或无效。跳过此影像的输出。")
                    continue

                # 从窗口读取数据。对于大文件，这只读取所需部分，显著减少内存消耗。
                data = src.read(window=window)

                # 转换为全色（灰度）图像
                # 最简单的方法是取所有波段的平均值
                if data.shape[0] > 1:
                    pan_data = np.mean(data, axis=0).astype(data.dtype)
                else:
                    pan_data = data[0, :, :] # 如果已经是单波段，直接使用

                # 更新输出文件的元数据
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": pan_data.shape[0],
                    "width": pan_data.shape[1],
                    "transform": src.window_transform(window), # 新窗口的地理变换信息
                    "count": 1, # 输出为单波段（全色）
                    "dtype": pan_data.dtype,
                    "compress": "LZW", # LZW 是常用的无损压缩
                    # 如果输出文件可能非常大（大于 4GB），启用 BigTIFF
                    "bigtiff": "YES" if pan_data.nbytes > 4 * (1024**3) else "NO" 
                })
                
                output_filename = os.path.join(output_dir, f"overlap_{i+1}_{os.path.basename(img_path)}")
                
                with rasterio.open(output_filename, "w", **out_meta) as dest:
                    dest.write(pan_data, 1) # 写入单波段数据

                print(f"  裁剪并保存了影像 {os.path.basename(img_path)} 的重叠部分到 {output_filename}")
        except rasterio.errors.RasterioIOError as e:
            print(f"错误：处理影像 {img_path} 时发生 IO 错误。错误信息：{e}")
        except Exception as e:
            print(f"处理影像 {img_path} 的重叠部分时发生未知错误：{e}")

# --- Configuration ---
MIN_OVERLAP_SIDE_METERS = 3000             # Minimum side length for the inscribed rectangle in meters

# --- Main Script ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    INPUT_FOLDER = os.path.join(options.root,'raw')
    OUTPUT_FOLDER = os.path.join(options.root,'pairs')

    # 确保输出目录存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. 获取所有 .tif 影像路径
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.tif'))
    if not image_paths:
        print(f"在 {INPUT_FOLDER} 中未找到任何 .tif 影像。程序退出。")
        exit()

    print(f"共找到 {len(image_paths)} 张影像。开始处理...")

    # 2. 提取所有影像的足迹多边形和 CRS
    image_data = {} # 存储 {路径: (足迹多边形, CRS)}
    for i, img_path in enumerate(image_paths):
        print(f"({i+1}/{len(image_paths)}) 正在提取影像足迹：{os.path.basename(img_path)}")
        footprint, crs = get_image_footprint_and_crs(img_path)
        if footprint is not None and crs is not None:
            image_data[img_path] = (footprint, crs)
        else:
            print(f"未能为影像 {os.path.basename(img_path)} 提取有效的足迹或 CRS。将跳过此影像。")

    # 3. 计算两两影像的重叠
    processed_pairs = set() # 用于避免重复处理 (A,B) 和 (B,A)
    overlap_counter = 0

    # 将字典键转换为列表，以便通过索引访问
    image_paths_list = list(image_data.keys()) 
    
    for i in range(len(image_paths_list)):
        for j in range(i + 1, len(image_paths_list)):
            img1_path = image_paths_list[i]
            img2_path = image_paths_list[j]

            # 使用排序后的路径元组作为键，确保对每对影像只处理一次
            pair_key = tuple(sorted((img1_path, img2_path)))
            if pair_key in processed_pairs:
                continue # 已处理过此对影像

            footprint1, crs1 = image_data[img1_path]
            footprint2, crs2 = image_data[img2_path]
            
            print(f"\n正在检查以下影像对的重叠情况：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")

            # 计算重叠并寻找满足条件的内接矩形
            inscribed_rect_original_crs = calculate_overlap_and_inscribed_rectangle(
                footprint1, crs1, footprint2, crs2, MIN_OVERLAP_SIDE_METERS
            )

            if inscribed_rect_original_crs:
                overlap_counter += 1
                print(f"✅ 找到符合要求的重叠区域 (第 {overlap_counter} 对) ：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")
                
                # 为这对影像创建一个独立的输出文件夹
                # 文件夹名包含序号和两张影像的文件名（不含扩展名）
                output_pair_dir = os.path.join(
                    OUTPUT_FOLDER,
                    f"overlap_pair_{overlap_counter}_{os.path.basename(os.path.splitext(img1_path)[0])}_{os.path.basename(os.path.splitext(img2_path)[0])}"
                )
                process_overlap(img1_path, img2_path, output_pair_dir, inscribed_rect_original_crs)
                processed_pairs.add(pair_key) # 将这对标记为已处理
            else:
                print(f"❌ 未找到符合要求的重叠区域：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")

    print(f"\n--- 处理完成 ---")
    print(f"总计找到并处理了 {overlap_counter} 对符合要求的影像重叠。")
    print(f"所有输出文件保存在：{OUTPUT_FOLDER}")

