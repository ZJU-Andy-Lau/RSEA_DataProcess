import rasterio
from rasterio.windows import Window
from shapely.geometry import box, Polygon
from shapely.ops import transform, unary_union
from rasterio.features import shapes
import pyproj
import numpy as np
import os
import argparse

def get_image_geometry(image_path):
    """
    获取遥感影像的几何信息。
    """
    with rasterio.open(image_path) as src:
        crs = src.crs
        transform_matrix = src.transform
        bounds = src.bounds
        width = src.width
        height = src.height
        return {'crs': crs, 'transform': transform_matrix, 'bounds': bounds, 'width': width, 'height': height}

def get_image_data_polygon(image_path):
    """
    获取遥感影像的有效数据区域的多边形。
    """
    with rasterio.open(image_path) as src:
        # Read the mask of the first band (usually 0 for NoData, 255 for valid data)
        mask = src.read_masks(1)

        # Iterate over shapes in the mask, extract polygons for valid data areas
        # transform=src.transform converts pixel coordinates to geographic coordinates
        image_polygons = []
        # Use a small transform for shape generation to handle potential floating point issues
        # and ensure polygons are properly closed.
        for geom, val in shapes(mask, transform=src.transform):
            if val == 255:  # 255 represents valid data area
                # For MultiPolygon parts, geom['coordinates'] is a list of lists of coordinates
                # For simple Polygon, it's a list of coordinates
                if geom['type'] == 'Polygon':
                    image_polygons.append(Polygon(geom['coordinates'][0]))
                elif geom['type'] == 'MultiPolygon':
                    for poly_coords in geom['coordinates']:
                        image_polygons.append(Polygon(poly_coords[0]))


        # Merge all valid data polygons into one (if the image has multiple disconnected valid areas)
        if image_polygons:
            # unary_union can merge multiple polygons, but for a single image, usually there's one large valid area.
            # If there are multiple, unary_union will create a MultiPolygon or a merged Polygon.
            return unary_union(image_polygons)
        else:
            return None

def calculate_overlap_metrics(geom1, geom2, poly1, poly2, target_crs_meters):
    """
    计算两幅影像的重叠区域，并以米为单位返回重叠区域的宽度和高度，以及精确的交集多边形。
    """
    # 1. 计算影像有效数据区域的精确交集
    if poly1 is None or poly2 is None:
        return 0, 0, None, None

    # Ensure the two polygons are in the same CRS for intersection calculation (assumed to be their original CRS here)
    exact_intersection_geom = poly1.intersection(poly2)

    if exact_intersection_geom.is_empty:
        return 0, 0, None, None

    # Ensure the intersection is a Polygon or MultiPolygon, not a GeometryCollection etc.
    # If it's a GeometryCollection, try to extract valid Polygons from it
    if exact_intersection_geom.geom_type == 'GeometryCollection':
        polygons_in_collection = [g for g in exact_intersection_geom.geoms if isinstance(g, Polygon)]
        if polygons_in_collection:
            exact_intersection_geom = unary_union(polygons_in_collection)
        else:
            return 0, 0, None, None # No valid Polygon intersection

    if exact_intersection_geom.is_empty: # Check again after potential extraction
        return 0, 0, None, None


    minx, miny, maxx, maxy = exact_intersection_geom.bounds

    # Convert the intersection polygon to the target metric CRS to calculate actual width and height
    project_to_meters = pyproj.Transformer.from_crs(geom1['crs'], target_crs_meters, always_xy=True).transform
    transformed_intersection = transform(project_to_meters, exact_intersection_geom)

    # Calculate bounds in meters
    m_minx, m_miny, m_maxx, m_maxy = transformed_intersection.bounds
    overlap_width_meters = m_maxx - m_minx
    overlap_height_meters = m_maxy - m_miny

    return overlap_width_meters, overlap_height_meters, exact_intersection_geom, box(minx, miny, maxx, maxy)


def process_image_overlaps(image_folder, output_folder, min_overlap_size_meters=(5000, 5000)):
    """
    处理文件夹中所有影像的重叠部分，筛选并输出符合要求的交集影像。
    """
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

    if not image_files:
        print("未找到任何tif格式的遥感影像。")
        return

    # Get geometric information and valid data polygons for all images
    image_geometries = {f: get_image_geometry(f) for f in image_files}
    image_data_polygons = {f: get_image_data_polygon(f) for f in image_files}

    target_crs_meters = pyproj.CRS("EPSG:32649") # Example: WGS 84 / UTM zone 49N

    processed_pairs = set()

    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            img1_path = image_files[i]
            img2_path = image_files[j]

            pair_key = tuple(sorted((img1_path, img2_path)))
            if pair_key in processed_pairs:
                continue

            geom1 = image_geometries[img1_path]
            geom2 = image_geometries[img2_path]
            poly1 = image_data_polygons[img1_path]
            poly2 = image_data_polygons[img2_path]

            print(f"正在计算 {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)} 的精确重叠...")
            overlap_width, overlap_height, exact_intersection_poly, clip_rect = \
                calculate_overlap_metrics(geom1, geom2, poly1, poly2, target_crs_meters)

            if overlap_width >= min_overlap_size_meters[0] and overlap_height >= min_overlap_size_meters[1] and clip_rect is not None:
                print(f"发现符合要求的精确重叠：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

                pair_output_dir = os.path.join(output_folder, f"{os.path.basename(img1_path).split('.tif')[0]}_{os.path.basename(img2_path).split('.tif')[0]}_overlap_aligned")
                os.makedirs(pair_output_dir, exist_ok=True)

                # Pass the bounds of clip_rect to the output function
                output_overlap_images(img1_path, img2_path, clip_rect.bounds, pair_output_dir)
                processed_pairs.add(pair_key)
            else:
                print(f"精确重叠尺寸不满足要求或不存在：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

def output_overlap_images(img1_path, img2_path, clip_bounds, output_dir):
    """
    输出两张影像的交集部分（基于clip_bounds），并将其转换为单波段全色影像（uint8类型）。
    clip_bounds: (minx, miny, maxx, maxy) 精确的裁剪地理边界
    """
    minx, miny, maxx, maxy = clip_bounds

    for img_path in [img1_path, img2_path]:
        with rasterio.open(img_path) as src:
            # Calculate the window in the current image corresponding to the intersection area
            window = src.window(minx, miny, maxx, maxy)

            # Convert the window to integer pixel coordinates, ensuring correct reading
            # Note: The window might be very small or have negative values due to floating-point precision; boundary checks are crucial.
            col_off = max(0, int(window.col_off))
            row_off = max(0, int(window.row_off))
            width = min(src.width - col_off, int(window.width))
            height = min(src.height - row_off, int(window.height))

            window_for_read = Window(col_off, row_off, width, height)

            if window_for_read.width <= 0 or window_for_read.height <= 0:
                print(f"警告: 影像 {os.path.basename(img_path)} 的交集窗口无效或过小，跳过输出。")
                continue
            
            # Ensure data is read using the correct window, considering large file optimization
            multispectral_data = None
            try:
                multispectral_data = src.read(window=window_for_read)
            except Exception as e:
                print(f"读取影像 {os.path.basename(img_path)} 的窗口数据时发生错误: {e}")
                print("尝试进行分块读取...")
                block_size = 2048 # Pixel block size for each read operation
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


            # --- Core modification to convert to panchromatic image: adding normalization to uint8 ---
            if multispectral_data is None or multispectral_data.size == 0:
                print(f"警告: 影像 {os.path.basename(img_path)} 没有有效数据被读取，跳过全色转换。")
                continue

            if multispectral_data.shape[0] > 1: # Ensure it's a multi-band image
                pan_data_float = np.mean(multispectral_data.astype(np.float32), axis=0)
            else:
                pan_data_float = multispectral_data[0].astype(np.float32)

            # Normalize to 0-255 range
            min_val = np.min(pan_data_float)
            max_val = np.max(pan_data_float)

            if max_val > min_val:
                pan_data_uint8 = ((pan_data_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                pan_data_uint8 = np.zeros_like(pan_data_float, dtype=np.uint8)
                if min_val > 0: # If value is not 0, set to maximum value 255
                    pan_data_uint8.fill(255)

            if pan_data_uint8.ndim == 3 and pan_data_uint8.shape[0] == 1:
                pan_data_uint8 = pan_data_uint8.squeeze()
            elif pan_data_uint8.ndim != 2:
                raise ValueError("Converted panchromatic data has incorrect dimensions; 2D expected.")

            # Create new transform matrix, now based on the precise clipping bounds
            out_transform = src.window_transform(window_for_read)


            output_filename = os.path.join(output_dir, f"{os.path.basename(img_path).split('.tif')[0]}_pan_aligned_uint8.tif")

            with rasterio.open(
                output_filename,
                'w',
                driver='GTiff',
                height=pan_data_uint8.shape[0],
                width=pan_data_uint8.shape[1],
                count=1,
                dtype=np.uint8,
                crs=src.crs,
                transform=out_transform, # Use the new transform matrix
                nodata=None               # NoData usually not retained after normalization
            ) as dst:
                dst.write(pan_data_uint8, 1)
        print(f"已生成精确对齐的全色交集影像 (uint8): {output_filename}")



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