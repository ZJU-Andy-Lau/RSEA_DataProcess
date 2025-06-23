import rasterio
from rasterio.windows import Window
from shapely.geometry import box, Polygon
from shapely.ops import transform, unary_union
from rasterio.features import shapes
import pyproj
import numpy as np
import os
import logging # 引入日志模块
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_geometry(image_path):
    """
    获取遥感影像的几何信息。
    """
    logging.info(f"正在获取影像几何信息: {os.path.basename(image_path)}")
    try:
        with rasterio.open(image_path) as src:
            crs = src.crs
            transform_matrix = src.transform
            bounds = src.bounds
            width = src.width
            height = src.height
        logging.info(f"已获取 {os.path.basename(image_path)} 的几何信息。")
        return {'crs': crs, 'transform': transform_matrix, 'bounds': bounds, 'width': width, 'height': height}
    except Exception as e:
        logging.error(f"获取影像 {os.path.basename(image_path)} 几何信息失败: {e}", exc_info=True)
        return None

def get_image_data_polygon(image_path):
    """
    获取遥感影像的有效数据区域的多边形。
    """
    logging.info(f"正在获取影像有效数据多边形: {os.path.basename(image_path)}")
    try:
        with rasterio.open(image_path) as src:
            # 检查是否有NoData值定义
            if src.nodata is None:
                logging.warning(f"影像 {os.path.basename(image_path)} 没有明确的NoData值，假定整个影像区域都是有效数据。")
                # 如果没有NoData，则其有效区域就是整个影像的边界框
                return box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

            # 否则，读取第一个波段的掩膜
            mask = src.read_masks(1)
            
            image_polygons = []
            # transform=src.transform 将像素坐标转换为地理坐标
            # precision=0 可能会提高大尺寸影像的处理速度，因为它不尝试构建非常精细的几何
            for geom, val in shapes(mask, transform=src.transform, connectivity=4):
                if val == 255:  # 255 表示有效数据区域
                    # 对于多边形或多部分多边形，geom['coordinates']的结构不同
                    if geom['type'] == 'Polygon':
                        # Polygon的坐标是一个列表的列表，第一个子列表是外边界
                        if geom['coordinates'] and geom['coordinates'][0]:
                            image_polygons.append(Polygon(geom['coordinates'][0]))
                    elif geom['type'] == 'MultiPolygon':
                        # MultiPolygon的坐标是多个列表的列表，每个代表一个Polygon的坐标
                        for poly_coords_list in geom['coordinates']:
                            if poly_coords_list and poly_coords_list[0]:
                                image_polygons.append(Polygon(poly_coords_list[0]))

        if image_polygons:
            # 合并所有有效数据多边形
            unified_polygon = unary_union(image_polygons)
            logging.info(f"已获取 {os.path.basename(image_path)} 的有效数据多边形。")
            return unified_polygon
        else:
            logging.warning(f"影像 {os.path.basename(image_path)} 未能提取到有效数据多边形。")
            return None
    except Exception as e:
        logging.error(f"获取影像 {os.path.basename(image_path)} 有效数据多边形失败: {e}", exc_info=True)
        return None

def calculate_overlap_metrics(geom1, geom2, poly1, poly2, target_crs_meters):
    """
    计算两幅影像的重叠区域，并以米为单位返回重叠区域的宽度和高度，以及精确的交集多边形。
    """
    logging.info(f"正在计算精确重叠指标...")
    if poly1 is None or poly2 is None:
        logging.warning("一个或多个影像的有效数据多边形为空，跳过重叠计算。")
        return 0, 0, None, None

    try:
        # 1. 计算影像有效数据区域的精确交集
        # Shapely的交集操作是基于笛卡尔坐标系的，所以确保CRS匹配很重要
        # 这里假设 poly1 和 poly2 已经在它们原始的地理CRS中
        exact_intersection_geom = poly1.intersection(poly2)

        if exact_intersection_geom.is_empty:
            logging.info("精确交集为空。")
            return 0, 0, None, None

        # 确保交集是 Polygon 或 MultiPolygon，而不是 GeometryCollection 等。
        if exact_intersection_geom.geom_type == 'GeometryCollection':
            polygons_in_collection = [g for g in exact_intersection_geom.geoms if isinstance(g, Polygon)]
            if polygons_in_collection:
                exact_intersection_geom = unary_union(polygons_in_collection)
            else:
                logging.warning("精确交集是一个空的 GeometryCollection。")
                return 0, 0, None, None

        if exact_intersection_geom.is_empty: # 潜在的二次检查
            logging.info("精确交集为空 (二次检查)。")
            return 0, 0, None, None

        # 2. 找到这个精确交集内的“最大内接矩形”
        # 我们使用精确交集的地理边界框(bounds)作为裁剪区域。
        minx, miny, maxx, maxy = exact_intersection_geom.bounds

        # 将交集多边形转换为目标米制CRS，计算实际宽度和高度
        # 这里使用 poly1 的CRS作为源CRS，假定poly1和poly2在同源CRS下
        project_to_meters = pyproj.Transformer.from_crs(geom1['crs'], target_crs_meters, always_xy=True).transform
        transformed_intersection = transform(project_to_meters, exact_intersection_geom)

        # 计算米制下的边界
        m_minx, m_miny, m_maxx, m_maxy = transformed_intersection.bounds
        overlap_width_meters = m_maxx - m_minx
        overlap_height_meters = m_maxy - m_miny

        logging.info(f"精确重叠计算完成，交集宽度: {overlap_width_meters:.2f}m, 高度: {overlap_height_meters:.2f}m")
        # 返回裁剪矩形的地理坐标 (minx, miny, maxx, maxy)
        return overlap_width_meters, overlap_height_meters, exact_intersection_geom, box(minx, miny, maxx, maxy)

    except Exception as e:
        logging.error(f"计算精确重叠指标失败: {e}", exc_info=True)
        return 0, 0, None, None


def process_image_overlaps(image_folder, output_folder, min_overlap_size_meters=(5000, 5000)):
    """
    处理文件夹中所有影像的重叠部分，筛选并输出符合要求的交集影像。
    """
    logging.info(f"开始处理影像文件夹: {image_folder}")
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

    if not image_files:
        logging.warning("未找到任何tif格式的遥感影像。")
        print("未找到任何tif格式的遥感影像。请检查 'your_image_folder' 路径是否正确且包含.tif文件。")
        return

    # 获取所有影像的几何信息和有效数据多边形
    image_geometries = {}
    image_data_polygons = {}
    for f in image_files:
        geom = get_image_geometry(f)
        if geom:
            image_geometries[f] = geom
            poly = get_image_data_polygon(f)
            if poly:
                image_data_polygons[f] = poly
            else:
                logging.error(f"无法获取影像 {os.path.basename(f)} 的有效数据多边形，将跳过涉及此影像的计算。")
        else:
            logging.error(f"无法获取影像 {os.path.basename(f)} 的几何信息，将跳过涉及此影像的计算。")

    # 筛选掉未能成功获取几何信息或多边形的影像
    valid_image_files = [f for f in image_files if f in image_geometries and f in image_data_polygons]
    if len(valid_image_files) < 2:
        logging.error("可用影像不足两张进行重叠计算。")
        print("可用影像不足两张进行重叠计算。请确保至少有两张有效的.tif影像。")
        return

    target_crs_meters = pyproj.CRS("EPSG:32649") # 示例：WGS 84 / UTM zone 49N

    processed_pairs = set()

    # 嵌套循环处理每对影像
    for i in range(len(valid_image_files)):
        for j in range(i + 1, len(valid_image_files)):
            img1_path = valid_image_files[i]
            img2_path = valid_image_files[j]

            pair_key = tuple(sorted((img1_path, img2_path)))
            if pair_key in processed_pairs:
                continue

            geom1 = image_geometries[img1_path]
            geom2 = image_geometries[img2_path]
            poly1 = image_data_polygons[img1_path]
            poly2 = image_data_polygons[img2_path]

            logging.info(f"正在处理影像对: {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)}")
            overlap_width, overlap_height, exact_intersection_poly, clip_rect = \
                calculate_overlap_metrics(geom1, geom2, poly1, poly2, target_crs_meters)

            if overlap_width >= min_overlap_size_meters[0] and \
               overlap_height >= min_overlap_size_meters[1] and \
               clip_rect is not None and not clip_rect.is_empty:
                
                logging.info(f"发现符合要求的精确重叠：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

                pair_output_dir = os.path.join(output_folder, f"{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}_overlap_aligned")
                os.makedirs(pair_output_dir, exist_ok=True)

                output_overlap_images(img1_path, img2_path, clip_rect.bounds, pair_output_dir)
                processed_pairs.add(pair_key)
            else:
                logging.info(f"影像对 {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)} 的精确重叠不满足要求或不存在。")
                print(f"影像对 {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)} 的重叠尺寸不满足要求或不存在：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

    logging.info("所有重叠计算和输出任务完成。")
    print("所有重叠计算和全色 (uint8) 输出任务完成！")


def output_overlap_images(img1_path, img2_path, clip_bounds, output_dir):
    """
    输出两张影像的交集部分（基于clip_bounds），并将其转换为单波段全色影像（uint8类型）。
    clip_bounds: (minx, miny, maxx, maxy) 精确的裁剪地理边界
    """
    minx, miny, maxx, maxy = clip_bounds
    logging.info(f"正在为 {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)} 输出交集影像到 {output_dir}")

    for img_path in [img1_path, img2_path]:
        try:
            with rasterio.open(img_path) as src:
                # 计算交集区域在当前影像中的窗口
                window = src.window(minx, miny, maxx, maxy)

                # 将窗口转换为整数像素坐标，确保读取正确
                col_off = max(0, int(window.col_off))
                row_off = max(0, int(window.row_off))
                width = min(src.width - col_off, int(window.width))
                height = min(src.height - row_off, int(window.height))

                window_for_read = Window(col_off, row_off, width, height)

                if window_for_read.width <= 0 or window_for_read.height <= 0:
                    logging.warning(f"影像 {os.path.basename(img_path)} 的裁剪窗口无效或过小 ({window_for_read.width}x{window_for_read.height})，跳过输出。")
                    continue
                
                multispectral_data = None
                # 判断是否需要分块读取
                # 经验法则：如果裁剪区域的像素数量太大，考虑分块
                # 例如，如果像素数量超过 100MB 估计值，就分块
                # (width * height * src.count * src.dtypes[0].itemsize) / (1024*1024) > 100MB
                estimated_memory_mb = (window_for_read.width * window_for_read.height * src.count * np.dtype(src.dtypes[0]).itemsize) / (1024*1024)

                if estimated_memory_mb > 500: # 假设500MB是一个阈值
                    logging.info(f"影像 {os.path.basename(img_path)} 裁剪区域过大 ({estimated_memory_mb:.2f}MB)，尝试分块读取。")
                    block_size = 2048
                    multispectral_data = np.empty((src.count, window_for_read.height, window_for_read.width), dtype=src.dtypes[0])

                    for b in range(src.count):
                        for r_off_block in range(0, window_for_read.height, block_size):
                            for c_off_block in range(0, window_for_read.width, block_size):
                                current_block_window = Window(window_for_read.col_off + c_off_block,
                                                              window_for_read.row_off + r_off_block,
                                                              min(block_size, window_for_read.width - c_off_block),
                                                              min(block_size, window_for_read.height - r_off_block))
                                if current_block_window.width > 0 and current_block_window.height > 0:
                                    block_data = src.read(b + 1, window=current_block_window)
                                    multispectral_data[b, r_off_block:r_off_block+block_data.shape[0], c_off_block:c_off_block+block_data.shape[1]] = block_data
                else:
                    multispectral_data = src.read(window=window_for_read)

                # --- 转换为全色影像并归一化到uint8 ---
                if multispectral_data is None or multispectral_data.size == 0:
                    logging.warning(f"影像 {os.path.basename(img_path)} 没有有效数据被读取，跳过全色转换。")
                    continue

                if multispectral_data.shape[0] > 1: # 确保是多波段影像
                    pan_data_float = np.mean(multispectral_data.astype(np.float32), axis=0)
                else:
                    pan_data_float = multispectral_data[0].astype(np.float32)

                # 归一化到 0-255 范围
                min_val = np.min(pan_data_float)
                max_val = np.max(pan_data_float)

                if max_val > min_val:
                    pan_data_uint8 = ((pan_data_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    pan_data_uint8 = np.zeros_like(pan_data_float, dtype=np.uint8)
                    if min_val > 0:
                        pan_data_uint8.fill(255)

                if pan_data_uint8.ndim == 3 and pan_data_uint8.shape[0] == 1:
                    pan_data_uint8 = pan_data_uint8.squeeze()
                elif pan_data_uint8.ndim != 2:
                    raise ValueError("转换后的全色数据维度不正确，预期为二维。")

                # 创建新的变换矩阵
                out_transform = src.window_transform(window_for_read)

                output_filename = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_pan_aligned_uint8.tif")

                with rasterio.open(
                    output_filename,
                    'w',
                    driver='GTiff',
                    height=pan_data_uint8.shape[0],
                    width=pan_data_uint8.shape[1],
                    count=1,
                    dtype=np.uint8,
                    crs=src.crs,
                    transform=out_transform,
                    nodata=None
                ) as dst:
                    dst.write(pan_data_uint8, 1)
            logging.info(f"已成功生成精确对齐的全色交集影像 (uint8): {output_filename}")
        except Exception as e:
            logging.error(f"处理影像 {os.path.basename(img_path)} 输出交集时发生错误: {e}", exc_info=True)



if __name__ == "__main__":
    # 示例用法
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    input_directory = os.path.join(options.root,'raw')
    output_directory_pairs = os.path.join(options.root,'pairs')
    # output_directory_triples = os.path.join(options.root,'triples')


    # 确保输出文件夹存在
    os.makedirs(output_directory_pairs,exist_ok=True)

    process_image_overlaps(input_directory,output_directory_pairs)
    print("所有重叠计算和输出任务完成！")