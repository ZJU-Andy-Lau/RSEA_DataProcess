import os
import glob
import rasterio
import rasterio.features
import rasterio.errors
import rasterio.warp # 修正：显式导入 rasterio.warp
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
                # 对于非常大的影像，read_masks() 可能仍然是内存密集型的。
                # 如果遇到内存问题，可能需要考虑分块处理掩膜或简化 NoData 处理。
                mask = src.read_masks(1) 
                
                # rasterio.features.shapes() 返回的是 (几何体, 值) 对的生成器
                # 只获取值为255（有效数据）的几何体
                valid_shapes = [
                    Polygon(geom['coordinates'][0]) for geom, val in rasterio.features.shapes(mask, transform=src.transform) if val == 255
                ]
                
                if not valid_shapes:
                    print(f"警告：影像 {os.path.basename(image_path)} 未找到有效数据形状（可能是全NoData）。将使用影像边界作为足迹。")
                else:
                    try:
                        # 尝试将所有有效数据多边形合并成一个单一的多边形或多重多边形
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

def calculate_overlap_and_inscribed_rectangle(poly1_orig_crs, crs1, poly2_orig_crs, crs2, min_side_meters):
    """
    计算两个多边形的重叠区域，找到重叠区域的轴对齐边界框，
    并检查其边长是否满足最小长度要求。
    如果符合要求，返回在 poly1 原始 CRS 下的轴对齐重叠矩形；否则返回 None。
    """
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

    try:
        poly1_proj = reproject_polygon(poly1_orig_crs, crs1, proj_crs)
        poly2_proj = reproject_polygon(poly2_orig_crs, crs2, proj_crs)
    except ValueError as e:
        print(f"重投影到 UTM CRS 失败: {e}. 跳过当前重叠计算。")
        return None

    intersection_proj = poly1_proj.intersection(poly2_proj)

    if intersection_proj.is_empty or not (isinstance(intersection_proj, Polygon) or isinstance(intersection_proj, unary_union)): 
        return None 

    # 找到重叠区域的轴对齐边界框作为“内接矩形”
    # 这是用于确定输出区域的矩形，其内部可能会有原始影像的NoData
    minx_int, miny_int, maxx_int, maxy_int = intersection_proj.bounds
    inscribed_rect_proj = box(minx_int, miny_int, maxx_int, maxy_int) 

    width_proj = inscribed_rect_proj.bounds[2] - inscribed_rect_proj.bounds[0]
    height_proj = inscribed_rect_proj.bounds[3] - inscribed_rect_proj.bounds[1] 
    
    if width_proj >= min_side_meters and height_proj >= min_side_meters:
        try:
            reproject_back_transformer = Transformer.from_crs(proj_crs, crs1, always_xy=True)
            # 返回这个矩形，作为我们最终裁剪和输出的公共地理范围
            return transform(reproject_back_transformer.transform, inscribed_rect_proj)
        except ValueError as e:
            print(f"重投影内接矩形到原始 CRS 失败: {e}")
            return None
    else:
        return None

def process_overlap(img1_path, img2_path, output_dir, common_output_bbox_poly):
    """
    处理一对重叠影像：将其裁剪到精确的公共重叠区域，转换为全色（灰度）图像，并保存输出。
    common_output_bbox_poly 是在 img1 CRS 下的重叠区域的轴对齐边界框。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确定目标输出影像的 CRS、变换和尺寸
    # 我们将以 img1 的 CRS 作为目标输出的 CRS，并以 common_output_bbox_poly 为其地理边界
    with rasterio.open(img1_path) as src1:
        target_crs = src1.crs
        # 计算目标输出的 transform 和 dimensions
        # 这里假设输出的像素大小与 src1 相同，这是为了保持分辨率一致性
        # 如果需要统一像素大小，需要根据 src1.res 计算
        pixel_width = src1.res[0]
        pixel_height = src1.res[1] # 注意：通常为负值
        
        # 根据 common_output_bbox_poly 计算输出的宽度和高度（像素）
        # common_output_bbox_poly.bounds 是 (minx, miny, maxx, maxy)
        output_width_meters = common_output_bbox_poly.bounds[2] - common_output_bbox_poly.bounds[0]
        output_height_meters = common_output_bbox_poly.bounds[3] - common_output_bbox_poly.bounds[1]

        target_width_pixels = int(np.ceil(output_width_meters / pixel_width))
        target_height_pixels = int(np.ceil(output_height_meters / abs(pixel_height))) # 像素高度总是正的

        # 计算目标输出的仿射变换
        # 左上角坐标是 common_output_bbox_poly 的 minx, maxy (因为pixel_height是负值，所以对应的是最大y值)
        target_transform = rasterio.transform.from_bounds(
            common_output_bbox_poly.bounds[0],  # minx
            common_output_bbox_poly.bounds[1],  # miny
            common_output_bbox_poly.bounds[2],  # maxx
            common_output_bbox_poly.bounds[3],  # maxy
            target_width_pixels, 
            target_height_pixels
        )

    # 循环处理两张影像
    for i, img_path in enumerate([img1_path, img2_path]):
        output_filename = os.path.join(output_dir, f"overlap_{i+1}_{os.path.basename(img_path)}")
        
        try:
            with rasterio.open(img_path) as src:
                # 1. 获取当前影像的有效足迹
                src_footprint, src_crs = get_image_footprint_and_crs(img_path)
                if src_footprint is None or src_crs is None:
                    print(f"警告：无法获取影像 {os.path.basename(img_path)} 的足迹，跳过其重叠处理。")
                    continue
                
                # 2. 将公共输出边界框重投影到当前影像的 CRS
                # 这样可以确保在当前影像的CRS下进行正确的空间操作
                reprojected_common_bbox_poly = reproject_polygon(common_output_bbox_poly, target_crs, src_crs)
                
                # 3. 计算当前影像有效足迹与重投影后的公共输出边界框的交集
                # 这个交集才是我们需要精确裁切的“有数据”区域
                exact_overlap_polygon_in_src_crs = src_footprint.intersection(reprojected_common_bbox_poly)

                if exact_overlap_polygon_in_src_crs.is_empty:
                    print(f"警告：影像 {os.path.basename(img_path)} 在共同输出区域内无有效数据。输出将全部为 NoData。")
                    # 创建一个全 NoData 的数组
                    pan_data = np.full((target_height_pixels, target_width_pixels), OUTPUT_NODATA_VALUE, dtype=src.dtypes[0])
                else:
                    # 4. 栅格化精确重叠多边形以创建掩膜
                    # shapes=[(geometry, value)]，value在这里是255表示有效数据
                    shapes_for_mask = [(exact_overlap_polygon_in_src_crs, 255)]
                    
                    # 栅格化掩膜，与源影像对齐
                    src_mask = rasterio.features.rasterize(
                        shapes=shapes_for_mask,
                        out_shape=(src.height, src.width),
                        transform=src.transform,
                        fill=0, # 填充0表示NoData
                        all_touched=True, # 确保边界像素也被包含
                        dtype='uint8'
                    )
                    
                    # 5. 读取源影像所有波段数据
                    src_data = src.read()
                    
                    # 6. 使用 rasterio.warp.reproject 进行重采样和裁剪
                    # target_data 是将要写入重采样结果的数组
                    target_data = np.zeros((src.count, target_height_pixels, target_width_pixels), dtype=src.dtypes[0])
                    
                    rasterio.warp.reproject(
                        source=src_data,
                        destination=target_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata, # 原始影像的 NoData 值
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        dst_nodata=OUTPUT_NODATA_VALUE, # 输出影像的 NoData 值
                        resampling=Resampling.nearest, # 重采样方法，可根据需求选择bilinear, cubic等
                        num_threads=os.cpu_count() or 1, # 利用多核CPU加速
                        band_weights=src_mask # 使用精确掩膜，只有掩膜为255的区域才会被重投影
                    )

                    # 7. 转换为全色（灰度）图像
                    if target_data.shape[0] > 1:
                        # 确保只对有效数据进行平均，NoData区域保持为OUTPUT_NODATA_VALUE
                        # 我们可以通过掩膜来做这个，或者直接在numpy操作时忽略nodata
                        # reproject函数已经处理了dst_nodata，所以直接平均即可
                        pan_data = np.mean(target_data, axis=0).astype(target_data.dtype)
                    else:
                        pan_data = target_data[0, :, :] # 如果已经是单波段

                # 更新输出文件的元数据
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": target_height_pixels,
                    "width": target_width_pixels,
                    "transform": target_transform,
                    "crs": target_crs,
                    "count": 1, # 输出为单波段（全色）
                    "dtype": pan_data.dtype,
                    "nodata": OUTPUT_NODATA_VALUE, # 明确设置输出的 NoData 值
                    "compress": "LZW", 
                    "bigtiff": "YES" if pan_data.nbytes > 4 * (1024**3) else "NO" 
                })
                
                with rasterio.open(output_filename, "w", **out_meta) as dest:
                    dest.write(pan_data, 1) # 写入单波段数据

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

    image_data = {} # 存储 {路径: (足迹多边形, CRS)}
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

            # calculate_overlap_and_inscribed_rectangle 返回的是重叠区域的轴对齐边界框（Polygon对象），
            # 且位于 img1 的原始 CRS 下。这将作为所有输出影像的统一裁剪区域。
            common_output_bbox_poly = calculate_overlap_and_inscribed_rectangle(
                footprint1, crs1, footprint2, crs2, MIN_OVERLAP_SIDE_METERS
            )

            if common_output_bbox_poly:
                overlap_counter += 1
                print(f"✅ 找到符合要求的重叠区域 (第 {overlap_counter} 对) ：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")
                
                output_pair_dir = os.path.join(
                    OUTPUT_FOLDER,
                    f"overlap_pair_{overlap_counter}_{os.path.basename(os.path.splitext(img1_path)[0])}_{os.path.basename(os.path.splitext(img2_path)[0])}"
                )
                # 将 common_output_bbox_poly 传入 process_overlap
                process_overlap(img1_path, img2_path, output_pair_dir, common_output_bbox_poly)
                processed_pairs.add(pair_key) 
            else:
                print(f"❌ 未找到符合要求的重叠区域：\n  - {os.path.basename(img1_path)}\n  - {os.path.basename(img2_path)}")

    print(f"\n--- 处理完成 ---")
    print(f"总计找到并处理了 {overlap_counter} 对符合要求的影像重叠。")
    print(f"所有输出文件保存在：{OUTPUT_FOLDER}")


