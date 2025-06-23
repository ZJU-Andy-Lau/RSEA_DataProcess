import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import numpy as np
import os
from itertools import combinations
import argparse
from pyproj import CRS, Transformer

def read_as_panchromatic(image_path, target_crs=None):
    """
    读取多光谱影像并将其转换为全色影像（简单地取所有波段的平均值）。
    如果提供了target_crs，则会将其重投影到目标CRS，同时保留原始CRS信息。
    """
    with rasterio.open(image_path) as src:
        original_profile = src.profile
        original_crs = src.crs

        if target_crs and src.crs != target_crs:
            print(f"Reprojecting {os.path.basename(image_path)} from {src.crs.to_string()} to {target_crs.to_string()} for processing.")
            # Calculate the transform and dimensions for the reprojected image
            _transform, _width, _height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            # Create an empty array for the reprojected data
            reprojected_data = np.zeros((src.count, _height, _width), dtype=src.profile['dtype'])

            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=reprojected_data[i-1, :, :],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest # Or other suitable resampling method
                )
            data = reprojected_data
            
            profile = original_profile.copy()
            profile.update({
                'crs': target_crs,
                'transform': _transform,
                'width': _width,
                'height': _height
            })
        else:
            data = src.read()
            profile = original_profile.copy()

        # 计算所有波段的平均值作为全色影像
        panchromatic_data = np.mean(data, axis=0).astype(profile['dtype'])
        
        # 更新profile
        profile.update(
            count=1,  # 全色影像只有一个波段
            dtype=panchromatic_data.dtype  # 更新数据类型
        )
        return panchromatic_data, profile, original_crs

def get_bounding_box(profile):
    """
    从影像的profile中获取其边界框 (minx, miny, maxx, maxy)。
    """
    transform = profile['transform']
    width = profile['width']
    height = profile['height']

    minx = transform.c
    maxy = transform.f
    maxx = transform.c + width * transform.a + height * transform.b
    miny = transform.f + width * transform.d + height * transform.e

    return minx, miny, maxx, maxy

def find_overlap(bbox1, bbox2, min_overlap_size_m, transform1, transform2):
    """
    计算两个边界框的重叠区域，并检查是否满足最小尺寸要求。
    返回重叠区域的地理坐标 (min_overlap_x, min_overlap_y, max_overlap_x, max_overlap_y)
    以及像素尺寸。
    假设输入的bbox是已经统一到以米为单位的CRS下的坐标。
    """
    minx1, miny1, maxx1, maxy1 = bbox1
    minx2, miny2, maxx2, maxy2 = bbox2

    # 计算重叠区域的地理坐标
    overlap_minx = max(minx1, minx2)
    overlap_miny = max(miny1, miny2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_maxy = min(maxy1, maxy2)

    # 检查是否有重叠
    if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
        return None, None

    # 计算重叠区域的宽度和高度（米）
    overlap_width_m = overlap_maxx - overlap_minx
    overlap_height_m = overlap_maxy - overlap_miny

    # 将米转换为像素
    # 由于不同影像可能分辨率不同，这里取第一个影像的像素尺寸进行粗略估计
    pixel_width_1 = abs(transform1.a)
    pixel_height_1 = abs(transform1.e)
    overlap_width_pixels = int(overlap_width_m / pixel_width_1)
    overlap_height_pixels = int(overlap_height_m / pixel_height_1)

    # 检查重叠区域尺寸是否满足要求
    if overlap_width_m >= min_overlap_size_m and overlap_height_m >= min_overlap_size_m:
        return (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy), (overlap_width_pixels, overlap_height_pixels)
    else:
        return None, None

def extract_and_save_overlap(image_path, overlap_bbox_meters, output_folder, output_suffix, original_profile_for_output, original_crs_for_output):
    """
    从原始影像中提取重叠区域并保存为新的全色GeoTIFF文件。
    overlap_bbox_meters 假定是以米为单位的CRS下的边界框。
    original_profile_for_output 和 original_crs_for_output 是为了保持输出影像与原始影像的CRS一致。
    """
    minx_m, miny_m, maxx_m, maxy_m = overlap_bbox_meters

    with rasterio.open(image_path) as src:
        # Check if the source CRS is different from the original_crs_for_output (which is the CRS the overlap_bbox is in)
        # If so, transform the overlap_bbox_meters back to the source CRS for correct windowing
        if src.crs != original_crs_for_output:
            transformer = Transformer.from_crs(original_crs_for_output, src.crs, always_xy=True)
            # transform_bounds handles min/max correctly after reprojection
            src_bounds_reprojected = transform_bounds(original_crs_for_output, src.crs, minx_m, miny_m, maxx_m, maxy_m)
            minx_src, miny_src, maxx_src, maxy_src = src_bounds_reprojected
        else:
            minx_src, miny_src, maxx_src, maxy_src = minx_m, miny_m, maxx_m, maxy_m


        # 使用窗口读取重叠区域数据
        # rasterio.windows.from_bounds 能够根据地理坐标计算出对应的像素窗口
        # Note: rasterio.window.from_bounds expects coordinates in the source CRS
        window = src.window(minx_src, miny_src, maxx_src, maxy_src)
        data = src.read(window=window)
        
        # 将多光谱数据转换为全色
        panchromatic_data = np.mean(data, axis=0).astype(original_profile_for_output['dtype'])

        # 更新profile以匹配重叠区域
        # calculate_default_transform also needs bounds in the source CRS
        transform, width, height = rasterio.windows.calculate_default_transform(
            src.crs, src.transform, src.width, src.height, window=window
        )
        
        output_profile = original_profile_for_output.copy()
        output_profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'count': 1,
            'dtype': panchromatic_data.dtype,
            'crs': src.crs # Ensure output CRS is the original image's CRS
        })

        output_filename = os.path.join(output_folder, f"{os.path.basename(image_path).replace('.tif', '')}_{output_suffix}.tif")
        with rasterio.open(output_filename, 'w', **output_profile) as dst:
            dst.write(panchromatic_data, 1)
        print(f"Saved overlapping region to: {output_filename}")


def process_images(input_folder, output_folder_pairs, output_folder_triples):
    """
    处理文件夹中的所有影像，查找两两和三三重叠区域。
    """
    if not os.path.exists(output_folder_pairs):
        os.makedirs(output_folder_pairs)
    if not os.path.exists(output_folder_triples):
        os.makedirs(output_folder_triples)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

    # 定义一个目标CRS，例如 UTM 区域的 CRS，或者一个常见的投影CRS，如 EPSG:3857 (Web Mercator)
    # 假设我们希望在计算重叠时使用以米为单位的投影CRS
    # 如果你的数据位于特定 UTM 区域，建议使用该区域的 CRS
    # 例如，如果你的研究区域在中国，你可以选择一个 UTM 区域的 CRS，如 EPSG:32650 (WGS 84 / UTM zone 50N)
    # 这里我们使用一个通用的投影CRS (Web Mercator)，它以米为单位
    target_meter_crs = CRS.from_epsg(3857) 
    print(f"Using {target_meter_crs.to_string()} (EPSG:{target_meter_crs.to_epsg()}) as the target CRS for overlap calculations.")

    # 存储影像信息： (全色数据, profile, 原始路径, 原始CRS)
    image_data = []
    for img_path in image_files:
        try:
            pan_data, profile, original_crs = read_as_panchromatic(img_path, target_crs=target_meter_crs)
            # 使用 .to_string() 或 .to_epsg() 替代 .name
            print(f"Successfully read and converted {os.path.basename(img_path)} to panchromatic and reprojected to {profile['crs'].to_string()}.")
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue
        image_data.append({'path': img_path, 'pan_data': pan_data, 'profile': profile, 'original_crs': original_crs})

    # --- 寻找两两重叠区域 ---
    print("\n--- Searching for pairwise overlapping regions (>= 5000m x 5000m) ---")
    processed_pairs = set() # To avoid processing the same pair twice (img1, img2) vs (img2, img1)
    for i in range(len(image_data)):
        for j in range(i + 1, len(image_data)):
            img1_info = image_data[i]
            img2_info = image_data[j]

            # Use a tuple of sorted paths to ensure uniqueness
            pair_key = tuple(sorted((img1_info['path'], img2_info['path'])))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            bbox1 = get_bounding_box(img1_info['profile']) # These bboxes are in target_meter_crs
            bbox2 = get_bounding_box(img2_info['profile']) # These bboxes are in target_meter_crs

            overlap_bbox_meters, overlap_pixels = find_overlap(bbox1, bbox2, 5000, img1_info['profile']['transform'], img2_info['profile']['transform'])

            if overlap_bbox_meters:
                print(f"Found significant overlap between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")
                print(f"  Overlap BBox (geographic, in {target_meter_crs.to_string()}): {overlap_bbox_meters}")
                print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

                # 提取并保存重叠区域，确保输出CRS与原始影像一致
                extract_and_save_overlap(img1_info['path'], overlap_bbox_meters, output_folder_pairs, f"overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}", img1_info['profile'], img1_info['original_crs'])
                extract_and_save_overlap(img2_info['path'], overlap_bbox_meters, output_folder_pairs, f"overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}", img2_info['profile'], img2_info['original_crs'])
            else:
                print(f"No significant overlap found between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")

    # --- 尝试寻找三张影像重叠区域 ---
    print("\n--- Searching for triple overlapping regions (>= 1000m x 1000m) ---")
    processed_triples = set()
    for img1_info, img2_info, img3_info in combinations(image_data, 3):
        # Create a unique key for the triple
        triple_key = tuple(sorted((img1_info['path'], img2_info['path'], img3_info['path'])))
        if triple_key in processed_triples:
            continue
        processed_triples.add(triple_key)

        # All bboxes are already in target_meter_crs due to read_as_panchromatic
        bboxes = [get_bounding_box(img1_info['profile']),
                  get_bounding_box(img2_info['profile']),
                  get_bounding_box(img3_info['profile'])]
        
        # Calculate the intersection of all three bounding boxes
        overlap_minx = max(bbox[0] for bbox in bboxes)
        overlap_miny = max(bbox[1] for bbox in bboxes)
        overlap_maxx = min(bbox[2] for bbox in bboxes)
        overlap_maxy = min(bbox[3] for bbox in bboxes)

        if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
            print(f"No triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            continue

        overlap_width_m = overlap_maxx - overlap_minx
        overlap_height_m = overlap_maxy - overlap_miny

        # For triple overlap, we'll use the first image's resolution for pixel estimation
        pixel_width_1 = abs(img1_info['profile']['transform'].a)
        pixel_height_1 = abs(img1_info['profile']['transform'].e)
        overlap_width_pixels = int(overlap_width_m / pixel_width_1)
        overlap_height_pixels = int(overlap_height_m / pixel_height_1)

        if overlap_width_m >= 1000 and overlap_height_m >= 1000:
            overlap_bbox_triple_meters = (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
            print(f"Found significant triple overlap for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            print(f"  Overlap BBox (geographic, in {target_meter_crs.to_string()}): {overlap_bbox_triple_meters}")
            print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

            # 提取并保存三张影像的重叠区域
            extract_and_save_overlap(img1_info['path'], overlap_bbox_triple_meters, output_folder_triples, f"triple_overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img1_info['profile'], img1_info['original_crs'])
            extract_and_save_overlap(img2_info['path'], overlap_bbox_triple_meters, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img2_info['profile'], img2_info['original_crs'])
            extract_and_save_overlap(img3_info['path'], overlap_bbox_triple_meters, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img2_info['path']).replace('.tif', '')}", img3_info['profile'], img3_info['original_crs'])
        else:
            print(f"No significant triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    input_directory = os.path.join(options.root,'raw')
    output_directory_pairs = os.path.join(options.root,'pairs')
    output_directory_triples = os.path.join(options.root,'triples')

    process_images(input_directory, output_directory_pairs, output_directory_triples)

    print("\n处理完成！请查看输出文件夹中的结果。")