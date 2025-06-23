import rasterio
import numpy as np
import os
from itertools import combinations
import argparse
from geopy.distance import geodesic
import warnings

# 忽略UserWarning，因为geopy在处理某些坐标时可能会发出关于精度的警告
warnings.filterwarnings("ignore", category=UserWarning)

def read_as_panchromatic_windowed(image_path, window=None):
    """
    读取多光谱影像的指定窗口，并将其转换为全色影像（简单地取所有波段的平均值）。
    如果未指定窗口，则读取整个影像。
    """
    with rasterio.open(image_path) as src:
        if window:
            data = src.read(window=window)
        else:
            data = src.read()
        
        # 检查是否所有波段都是空白（例如，在窗口外读取时）
        if data.size == 0:
            return None, None

        # 计算所有波段的平均值作为全色影像
        panchromatic_data = np.mean(data, axis=0).astype(src.profile['dtype'])
        
        # 更新profile以匹配窗口（如果指定了窗口）
        if window:
            transform, width, height = rasterio.windows.calculate_default_transform(
                src.crs, src.transform, src.width, src.height, window=window
            )
            profile = src.profile.copy()
            profile.update({
                'height': height,
                'width': width,
                'transform': transform,
                'count': 1,  # 全色影像只有一个波段
                'dtype': panchromatic_data.dtype
            })
        else:
            profile = src.profile.copy()
            profile.update(
                count=1,  # 全色影像只有一个波段
                dtype=panchromatic_data.dtype  # 更新数据类型
            )
        return panchromatic_data, profile

def get_bounding_box(profile):
    """
    从影像的profile中获取其边界框 (minx, miny, maxx, maxy)。
    """
    # rasterio的bounds属性直接提供了边界框
    bounds = profile['bounds']
    return bounds.left, bounds.bottom, bounds.right, bounds.top

def calculate_overlap_size_m(bbox1, bbox2, crs1, crs2):
    """
    计算两个边界框的重叠区域在地理上的宽度和高度（米），
    根据CRS类型选择不同的距离计算方法。
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
        return None, None, None

    # 获取重叠区域的中心点（用于距离估算）和角点
    center_x = (overlap_minx + overlap_maxx) / 2
    center_y = (overlap_miny + overlap_maxy) / 2

    # 判断CRS是否是地理坐标系
    is_geographic1 = crs1.is_geographic if crs1 else False
    is_geographic2 = crs2.is_geographic if crs2 else False

    # 这里我们假设如果两张影像的CRS都可用且都是地理坐标系，则使用测地线距离
    # 否则，如果其中一个不是地理坐标系或者不可用，则按投影坐标系处理（直接计算差值）
    # 实际应用中，如果CRS不匹配且其中一个是投影，可能需要更复杂的逻辑或警告用户
    if is_geographic1 and is_geographic2:
        # 均为地理坐标系，使用测地线距离计算宽度和高度
        # 计算宽度 (在中心纬度上的经度距离)
        p1_width = (overlap_miny, overlap_minx)
        p2_width = (overlap_miny, overlap_maxx)
        overlap_width_m = geodesic(p1_width, p2_width).meters

        # 计算高度 (在中心经度上的纬度距离)
        p1_height = (overlap_miny, overlap_minx)
        p2_height = (overlap_maxy, overlap_minx)
        overlap_height_m = geodesic(p1_height, p2_height).meters
    else:
        # 假设是投影坐标系或无法判断，直接计算差值
        # 警告：这里假设单位是米，如果不是米，则结果不准确
        # 在实际应用中，如果CRS是英尺等，需要进行单位转换
        print("警告：CRS不是地理坐标系或无法识别，将直接计算坐标差值作为距离，请确保单位是米。")
        overlap_width_m = overlap_maxx - overlap_minx
        overlap_height_m = overlap_maxy - overlap_miny

    # 将重叠区域的地理坐标转换为 (minx, miny, maxx, maxy) 格式
    overlap_bbox = (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
    return overlap_bbox, overlap_width_m, overlap_height_m

def find_overlap(bbox1, bbox2, min_overlap_size_m, profile1, profile2):
    """
    计算两个边界框的重叠区域，并检查是否满足最小尺寸要求。
    返回重叠区域的地理坐标 (min_overlap_x, min_overlap_y, max_overlap_x, max_overlap_y)
    以及像素尺寸。
    """
    overlap_bbox, overlap_width_m, overlap_height_m = calculate_overlap_size_m(bbox1, bbox2, profile1['crs'], profile2['crs'])

    if overlap_bbox is None:
        return None, None

    # 计算重叠区域的像素尺寸，使用第一个影像的分辨率进行估算
    # 注意：这里仅为估算，实际提取时rasterio会根据实际情况调整
    pixel_width_1 = abs(profile1['transform'].a)
    pixel_height_1 = abs(profile1['transform'].e)
    
    # 转换为像素尺寸时，需要考虑地理坐标系和投影坐标系的区别
    # 对于地理坐标系，简单除以度每像素的值不准确，这里只是一个粗略的像素估算
    # 实际在extract_and_save_overlap中，rasterio会根据地理范围和影像的transform来确定窗口
    
    # 尝试将米转换为像素，这是一个近似值，因为地理坐标系下像素不是等距的
    # 对于地理坐标系，我们无法直接用米来计算像素，但可以在 extract_and_save_overlap 中通过 `src.window(minx, miny, maxx, maxy)` 让 rasterio 自动处理
    # 这里的像素估算对于投影坐标系是准确的，对于地理坐标系则是一个粗略参考
    if profile1['crs'].is_geographic:
        # 对于地理坐标系，计算 bbox 经纬度差值与影像总经纬度差值的比例，再乘以像素数
        img1_total_width_deg = profile1['bounds'].right - profile1['bounds'].left
        img1_total_height_deg = profile1['bounds'].top - profile1['bounds'].bottom
        
        overlap_width_pixels = int((overlap_bbox[2] - overlap_bbox[0]) / img1_total_width_deg * profile1['width']) if img1_total_width_deg != 0 else 0
        overlap_height_pixels = int((overlap_bbox[3] - overlap_bbox[1]) / img1_total_height_deg * profile1['height']) if img1_total_height_deg != 0 else 0
    else:
        overlap_width_pixels = int(overlap_width_m / pixel_width_1)
        overlap_height_pixels = int(overlap_height_m / pixel_height_1)

    # 检查重叠区域尺寸是否满足要求
    if overlap_width_m >= min_overlap_size_m and overlap_height_m >= min_overlap_size_m:
        return overlap_bbox, (overlap_width_pixels, overlap_height_pixels)
    else:
        return None, None

def extract_and_save_overlap(image_path, overlap_bbox, output_folder, output_suffix, original_profile):
    """
    从原始影像中提取重叠区域并保存为新的全色GeoTIFF文件。
    """
    minx, miny, maxx, maxy = overlap_bbox

    with rasterio.open(image_path) as src:
        # 使用窗口读取重叠区域数据
        # rasterio.windows.from_bounds 能够根据地理坐标计算出对应的像素窗口
        # 对于大影像，这种方式只加载需要的部分，大大节省内存
        window = src.window(minx, miny, maxx, maxy)
        data = src.read(window=window)
        
        if data.size == 0:
            print(f"Warning: No data found for specified window in {os.path.basename(image_path)}. Skipping save.")
            return
            
        # 将多光谱数据转换为全色
        panchromatic_data = np.mean(data, axis=0).astype(original_profile['dtype'])

        # 更新profile以匹配重叠区域
        transform, width, height = rasterio.windows.calculate_default_transform(
            src.crs, src.transform, src.width, src.height, window=window
        )
        
        output_profile = original_profile.copy()
        output_profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'count': 1,
            'dtype': panchromatic_data.dtype
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

    # 存储影像信息： (profile, 原始路径)
    # 我们不再提前读取全部数据，而是在需要时按窗口读取
    image_info = []
    for img_path in image_files:
        try:
            with rasterio.open(img_path) as src:
                image_info.append({'path': img_path, 'profile': src.profile.copy()})
            print(f"Successfully read metadata for {os.path.basename(img_path)}.")
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading metadata for {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue

    # --- 寻找两两重叠区域 ---
    print("\n--- Searching for pairwise overlapping regions (>= 3000m x 3000m) ---")
    processed_pairs = set() # To avoid processing the same pair twice (img1, img2) vs (img2, img1)
    for i in range(len(image_info)):
        for j in range(i + 1, len(image_info)):
            img1_data = image_info[i]
            img2_data = image_info[j]

            # Use a tuple of sorted paths to ensure uniqueness
            pair_key = tuple(sorted((img1_data['path'], img2_data['path'])))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            bbox1 = get_bounding_box(img1_data['profile'])
            bbox2 = get_bounding_box(img2_data['profile'])

            overlap_bbox, overlap_width_m, overlap_height_m = calculate_overlap_size_m(
                bbox1, bbox2, img1_data['profile']['crs'], img2_data['profile']['crs']
            )

            if overlap_bbox and overlap_width_m >= 3000 and overlap_height_m >= 3000:
                print(f"Found significant overlap between {os.path.basename(img1_data['path'])} and {os.path.basename(img2_data['path'])}")
                print(f"  Overlap BBox (geographic): {overlap_bbox}")
                print(f"  Approximate Overlap Size (m): {overlap_width_m:.2f}m x {overlap_height_m:.2f}m")

                # 提取并保存重叠区域
                extract_and_save_overlap(img1_data['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img2_data['path']).replace('.tif', '')}", img1_data['profile'])
                extract_and_save_overlap(img2_data['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img1_data['path']).replace('.tif', '')}", img2_data['profile'])
            else:
                print(f"No significant overlap found between {os.path.basename(img1_data['path'])} and {os.path.basename(img2_data['path'])}")

    # --- 尝试寻找三张影像重叠区域 ---
    print("\n--- Searching for triple overlapping regions (>= 1000m x 1000m) ---")
    processed_triples = set()
    for img1_info, img2_info, img3_info in combinations(image_info, 3):
        # Create a unique key for the triple
        triple_key = tuple(sorted((img1_info['path'], img2_info['path'], img3_info['path'])))
        if triple_key in processed_triples:
            continue
        processed_triples.add(triple_key)

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

        # 将三张影像的 CRS 都传递给距离计算函数
        # 假设只要有一个是地理坐标系就用测地线距离，否则用直角坐标
        # 实际更严谨的做法是检查所有CRS是否一致，或者强制转换为通用投影
        crs_list = [img1_info['profile']['crs'], img2_info['profile']['crs'], img3_info['profile']['crs']]
        
        # 为了简化，这里我们仍然用前两张影像的CRS来决定距离计算方式
        # 更严谨的做法是检查是否所有CRS都是地理坐标系，或者统一转换到米单位的投影
        # 这里假设如果任意一个 CRS 是地理坐标系，就使用测地线距离来计算重叠区域的尺寸
        # 否则，就按直角坐标计算（假设单位是米）
        is_any_geographic = any(crs.is_geographic for crs in crs_list if crs)

        if is_any_geographic:
            # 均为地理坐标系，使用测地线距离计算宽度和高度
            p1_width = (overlap_miny, overlap_minx)
            p2_width = (overlap_miny, overlap_maxx)
            overlap_width_m = geodesic(p1_width, p2_width).meters

            p1_height = (overlap_miny, overlap_minx)
            p2_height = (overlap_maxy, overlap_minx)
            overlap_height_m = geodesic(p1_height, p2_height).meters
        else:
            print("警告：三张影像的CRS都不是地理坐标系或无法识别，将直接计算坐标差值作为距离，请确保单位是米。")
            overlap_width_m = overlap_maxx - overlap_minx
            overlap_height_m = overlap_maxy - overlap_miny

        if overlap_width_m >= 1000 and overlap_height_m >= 1000:
            overlap_bbox_triple = (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
            print(f"Found significant triple overlap for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            print(f"  Overlap BBox (geographic): {overlap_bbox_triple}")
            print(f"  Approximate Overlap Size (m): {overlap_width_m:.2f}m x {overlap_height_m:.2f}m")

            # 提取并保存三张影像的重叠区域
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