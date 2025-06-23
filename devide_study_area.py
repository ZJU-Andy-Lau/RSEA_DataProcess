import rasterio
import numpy as np
import os
from itertools import combinations
from rasterio.windows import Window
from rasterio.transform import Affine
import argparse
# from pyproj import CRS, Transformer # pyproj 用于更复杂的CRS转换，此处可能不需要直接使用Transformer，rasterio内部会处理

# 目标投影坐标系，例如 Web Mercator
TARGET_CRS = 'EPSG:3857' # 或者 'EPSG:326XX' (UTM Zone XXN), 'EPSG:327XX' (UTM Zone XXS)

def read_and_reproject_as_panchromatic(image_path, target_crs=TARGET_CRS):
    """
    读取多光谱影像，将其转换为全色影像，并重投影到指定CRS。
    """
    with rasterio.open(image_path) as src:
        # 如果原始CRS与目标CRS不同，则进行重投影
        if src.crs != target_crs:
            print(f"Reprojecting {os.path.basename(image_path)} from {src.crs} to {target_crs}...")
            # 计算重投影后的新尺寸和变换
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            # 创建一个新的profile用于重投影后的数据
            reprojected_profile = src.profile.copy()
            reprojected_profile.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'dtype': src.profile['dtype'] # 保持原始数据类型
            })

            # 创建一个内存中的数组来存储重投影后的数据
            reprojected_data = np.empty((src.count, height, width), dtype=src.profile['dtype'])

            # 执行重投影
            rasterio.warp.reproject(
                source=rasterio.band(src, range(1, src.count + 1)), # 读取所有波段
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=rasterio.enums.Resampling.nearest # 可以根据需求选择其他重采样方法，如bilinear, cubic
            )
            # 计算全色影像
            panchromatic_data = np.mean(reprojected_data, axis=0).astype(reprojected_profile['dtype'])
            reprojected_profile.update(count=1) # 全色影像只有一个波段

            return panchromatic_data, reprojected_profile
        else:
            # 如果CRS已经是目标CRS，则直接读取并转换为全色
            print(f"{os.path.basename(image_path)} already in {target_crs}, no reprojecting needed.")
            data = src.read()
            panchromatic_data = np.mean(data, axis=0).astype(src.profile['dtype'])
            profile = src.profile
            profile.update(count=1, dtype=panchromatic_data.dtype)
            return panchromatic_data, profile


# 保持 get_bounding_box 不变，因为它现在将处理米制坐标的 profile
def get_bounding_box(profile):
    """
    从影像的profile中获取其边界框 (minx, miny, maxx, maxy)。
    """
    transform = profile['transform']
    width = profile['width']
    height = profile['height']

    # 注意：这里的 transform.a 和 transform.e 现在应该是米/像素
    minx = transform.c
    maxy = transform.f
    maxx = transform.c + width * transform.a + height * transform.b
    miny = transform.f + width * transform.d + height * transform.e

    return minx, miny, maxx, maxy

# find_overlap 函数也不变，因为它现在接收米制单位的 bbox
def find_overlap(bbox1, bbox2, min_overlap_size_m, transform1, transform2):
    """
    计算两个边界框的重叠区域，并检查是否满足最小尺寸要求。
    返回重叠区域的地理坐标 (min_overlap_x, min_overlap_y, max_overlap_x, max_overlap_y)
    以及像素尺寸。
    """
    minx1, miny1, maxx1, maxy1 = bbox1
    minx2, miny2, maxx2, maxy2 = bbox2

    overlap_minx = max(minx1, minx2)
    overlap_miny = max(miny1, miny2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_maxy = min(maxy1, maxy2)

    if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
        return None, None

    overlap_width_m = overlap_maxx - overlap_minx
    overlap_height_m = overlap_maxy - overlap_miny

    # 这里的 pixel_width_1 已经是米/像素了
    pixel_width_1 = abs(transform1.a)
    pixel_height_1 = abs(transform1.e)
    overlap_width_pixels = int(overlap_width_m / pixel_width_1)
    overlap_height_pixels = int(overlap_height_m / pixel_height_1)

    if overlap_width_m >= min_overlap_size_m and overlap_height_m >= min_overlap_size_m:
        return (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy), (overlap_width_pixels, overlap_height_pixels)
    else:
        return None, None


# extract_and_save_overlap 保持不变，因为它将基于重投影后的 profile 进行裁剪和保存
def extract_and_save_overlap(image_path, overlap_bbox, output_folder, output_suffix, original_profile_after_reprojection):
    """
    从原始影像中提取重叠区域并保存为新的全色GeoTIFF文件。
    注意：这里的 image_path 仍然是原始路径，但是提取时我们根据重投影后的 profile 来确定地理坐标。
    为了正确裁剪，我们需要再次打开原始文件，然后根据计算出的重叠区域的地理坐标来确定裁剪窗口。
    理想情况下，我们应该对原始影像进行一次性重投影，然后基于重投影后的影像数据进行操作。
    下面的实现会再次打开原始影像，并根据 overlap_bbox (已转换为目标CRS的米制坐标) 来计算裁剪窗口。
    
    更优化的方式是：在 process_images 中存储重投影后的 pan_data 和 profile，
    然后 extract_and_save_overlap 直接使用这些重投影后的数据。
    但为了简化，我们在这里让它再次读取原始影像，并利用 rasterio 的强大功能直接从原始影像中裁剪重投影后的区域。
    """
    minx_overlap, miny_overlap, maxx_overlap, maxy_overlap = overlap_bbox

    with rasterio.open(image_path) as src:
        # 如果原始影像的CRS与用于计算 overlap_bbox 的CRS不同，需要转换 overlap_bbox 的坐标
        # rasterio.windows.from_bounds 会自动处理CRS转换，如果它们不同
        # 但为了确保准确性，最好是原始影像也重投影到 TARGET_CRS 再计算 window
        # 或者，更简单和鲁棒的方法是：将原始影像的范围投影到目标CRS，然后计算重叠
        # 这里为了简化，我们假设 src.crs 在这里要么就是原始的地理CRS，要么就是目标CRS。
        # rasterio.windows.from_bounds 内部会尝试根据 CRS 转换坐标，但最好是显式地保持一致。

        # 核心逻辑：定义窗口的地理边界
        # 这里的 window 是基于重投影后的目标CRS的地理边界来确定的
        # 如果 src.crs 与 TARGET_CRS 不同，rasterio 会在 from_bounds 内部进行坐标转换
        window = src.window(minx_overlap, miny_overlap, maxx_overlap, maxy_overlap)
        
        # 读取指定窗口的数据
        data = src.read(window=window)
        
        # 将多光谱数据转换为全色
        panchromatic_data = np.mean(data, axis=0).astype(original_profile_after_reprojection['dtype'])

        # 更新profile以匹配重叠区域
        # calculate_default_transform 将确保新的 transform 和尺寸是基于重叠区域在目标CRS中的
        transform, width, height = rasterio.windows.calculate_default_transform(
            src.crs, src.transform, src.width, src.height, window=window
        )
        
        # 确保输出的CRS是我们的目标CRS
        output_profile = original_profile_after_reprojection.copy()
        output_profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'count': 1,
            'dtype': panchromatic_data.dtype,
            'crs': TARGET_CRS # 确保输出的CRS是目标CRS
        })

        output_filename = os.path.join(output_folder, f"{os.path.basename(image_path).replace('.tif', '')}_{output_suffix}.tif")
        with rasterio.open(output_filename, 'w', **output_profile) as dst:
            dst.write(panchromatic_data, 1)
        print(f"Saved overlapping region to: {output_filename}")


def process_images(input_folder, output_folder_pairs, output_folder_triples):
    """
    处理文件夹中的所有影像，查找两两和三三重叠区域。
    现在在读取时会先进行重投影。
    """
    if not os.path.exists(output_folder_pairs):
        os.makedirs(output_folder_pairs)
    if not os.path.exists(output_folder_triples):
        os.makedirs(output_folder_triples)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

    image_data = []
    for img_path in image_files:
        try:
            # 调用新的重投影函数
            pan_data, profile = read_and_reproject_as_panchromatic(img_path, TARGET_CRS)
            image_data.append({'path': img_path, 'pan_data': pan_data, 'profile': profile})
            print(f"Processed {os.path.basename(img_path)} with CRS {profile['crs']}.")
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue

    # --- 寻找两两重叠区域 ---
    print("\n--- Searching for pairwise overlapping regions (>= 5000m x 5000m) ---")
    processed_pairs = set()
    for i in range(len(image_data)):
        for j in range(i + 1, len(image_data)):
            img1_info = image_data[i]
            img2_info = image_data[j]

            pair_key = tuple(sorted((img1_info['path'], img2_info['path'])))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # bbox 现在是米制单位的
            bbox1 = get_bounding_box(img1_info['profile'])
            bbox2 = get_bounding_box(img2_info['profile'])

            # transform1 和 transform2 也是重投影后的米制单位变换
            overlap_bbox, overlap_pixels = find_overlap(bbox1, bbox2, 5000, img1_info['profile']['transform'], img2_info['profile']['transform'])

            if overlap_bbox:
                print(f"Found significant overlap between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")
                print(f"  Overlap BBox (geographic, {TARGET_CRS}): {overlap_bbox}")
                print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

                # 提取并保存重叠区域，传入重投影后的 profile
                extract_and_save_overlap(img1_info['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}", img1_info['profile'])
                extract_and_save_overlap(img2_info['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}", img2_info['profile'])
            else:
                print(f"No significant overlap found between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")

    # --- 尝试寻找三张影像重叠区域 ---
    print("\n--- Searching for triple overlapping regions (>= 1000m x 1000m) ---")
    processed_triples = set()
    for img1_info, img2_info, img3_info in combinations(image_data, 3):
        triple_key = tuple(sorted((img1_info['path'], img2_info['path'], img3_info['path'])))
        if triple_key in processed_triples:
            continue
        processed_triples.add(triple_key)

        # bboxes 现在是米制单位的
        bboxes = [get_bounding_box(img1_info['profile']),
                  get_bounding_box(img2_info['profile']),
                  get_bounding_box(img3_info['profile'])]
        
        overlap_minx = max(bbox[0] for bbox in bboxes)
        overlap_miny = max(bbox[1] for bbox in bboxes)
        overlap_maxx = min(bbox[2] for bbox in bboxes)
        overlap_maxy = min(bbox[3] for bbox in bboxes)

        if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
            print(f"No triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            continue

        overlap_width_m = overlap_maxx - overlap_minx
        overlap_height_m = overlap_maxy - overlap_miny

        # 这里的 pixel_width_1 已经是米/像素了
        pixel_width_1 = abs(img1_info['profile']['transform'].a)
        pixel_height_1 = abs(img1_info['profile']['transform'].e)
        overlap_width_pixels = int(overlap_width_m / pixel_width_1)
        overlap_height_pixels = int(overlap_height_m / pixel_height_1)

        if overlap_width_m >= 1000 and overlap_height_m >= 1000:
            overlap_bbox_triple = (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
            print(f"Found significant triple overlap for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            print(f"  Overlap BBox (geographic, {TARGET_CRS}): {overlap_bbox_triple}")
            print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

            extract_and_save_overlap(img1_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img1_info['profile'])
            extract_and_save_overlap(img2_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img2_info['profile'])
            extract_and_save_overlap(img3_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img2_info['path']).replace('.tif', '')}", img3_info['profile'])
        else:
            print(f"No significant triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")


if __name__ == "__main__":
    # 配置输入和输出文件夹
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    input_directory = os.path.join(options.root,'raw')
    output_directory_pairs = os.path.join(options.root,'pairs')
    output_directory_triples = os.path.join(options.root,'triples')

    process_images(input_directory, output_directory_pairs, output_directory_triples)

    print("\n处理完成！请查看输出文件夹中的结果。")