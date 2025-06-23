import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import transform
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

def calculate_overlap_area(geom1, geom2, target_crs_meters):
    """
    计算两幅影像的重叠区域，并以米为单位返回重叠区域的宽度和高度。
    """
    # 将两个几何对象转换为 Shapely box
    bbox1 = box(geom1['bounds'].left, geom1['bounds'].bottom, geom1['bounds'].right, geom1['bounds'].top)
    bbox2 = box(geom2['bounds'].left, geom2['bounds'].bottom, geom2['bounds'].right, geom2['bounds'].top)

    # 计算交集
    intersection = bbox1.intersection(bbox2)

    if intersection.is_empty:
        return 0, 0, None

    # 定义转换函数，将地理坐标转换为目标米制CRS
    project_to_meters = pyproj.Transformer.from_crs(geom1['crs'], target_crs_meters, always_xy=True).transform

    # 将交集区域转换为米制CRS
    transformed_intersection = transform(project_to_meters, intersection)

    # 计算重叠区域的宽度和高度（米）
    minx, miny, maxx, maxy = transformed_intersection.bounds
    overlap_width_meters = maxx - minx
    overlap_height_meters = maxy - miny

    return overlap_width_meters, overlap_height_meters, intersection

def process_image_overlaps(image_folder, output_folder, min_overlap_size_meters=(5000, 5000)):
    """
    处理文件夹中所有影像的重叠部分，筛选并输出符合要求的交集影像。
    """
    # 获取所有tif影像文件
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

    if not image_files:
        print("未找到任何tif格式的遥感影像。")
        return

    # 获取所有影像的几何信息
    image_geometries = {f: get_image_geometry(f) for f in image_files}

    # 创建一个通用的米制CRS用于距离计算 (例如，UTM)
    # 假设所有影像都在同一个半球，或者选择一个适合大部分区域的CRS
    # 这里我们使用一个通用的UTM区域，实际应用中可能需要根据影像的具体位置选择
    target_crs_meters = pyproj.CRS("EPSG:32649") # 示例：WGS 84 / UTM zone 49N

    processed_pairs = set() # 记录已处理的影像对，避免重复计算和处理

    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            img1_path = image_files[i]
            img2_path = image_files[j]

            # 避免重复处理同一对影像（顺序不同）
            pair_key = tuple(sorted((img1_path, img2_path)))
            if pair_key in processed_pairs:
                continue

            geom1 = image_geometries[img1_path]
            geom2 = image_geometries[img2_path]

            print(f"正在计算 {os.path.basename(img1_path)} 和 {os.path.basename(img2_path)} 的重叠...")
            overlap_width, overlap_height, intersection_geom = calculate_overlap_area(geom1, geom2, target_crs_meters)

            if overlap_width >= min_overlap_size_meters[0] and overlap_height >= min_overlap_size_meters[1]:
                print(f"发现符合要求的重叠：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

                # 创建输出文件夹
                pair_output_dir = os.path.join(output_folder, f"{os.path.basename(img1_path).split('.tif')[0]}_{os.path.basename(img2_path).split('.tif')[0]}_overlap")
                os.makedirs(pair_output_dir, exist_ok=True)

                # 输出交集部分
                output_overlap_images(img1_path, img2_path, intersection_geom, pair_output_dir)
                processed_pairs.add(pair_key)
            else:
                print(f"重叠尺寸不满足要求：宽度 {overlap_width:.2f}m, 高度 {overlap_height:.2f}m")

def output_overlap_images(img1_path, img2_path, intersection_geom, output_dir):
    """
    输出两张影像的交集部分，并将其转换为单波段全色影像（uint8类型）。
    """
    minx, miny, maxx, maxy = intersection_geom.bounds

    for img_path in [img1_path, img2_path]:
        with rasterio.open(img_path) as src:
            window = src.window(minx, miny, maxx, maxy)

            window = Window(max(0, int(window.col_off)),
                            max(0, int(window.row_off)),
                            min(src.width - window.col_off, int(window.width)),
                            min(src.height - window.row_off, int(window.height)))

            if window.width <= 0 or window.height <= 0:
                print(f"警告: 影像 {os.path.basename(img_path)} 的交集窗口无效，跳过输出。")
                continue

            try:
                # 读取所有波段数据
                multispectral_data = src.read(window=window)
            except Exception as e:
                print(f"读取影像 {os.path.basename(img_path)} 的窗口数据时发生错误: {e}")
                print("尝试进行分块读取...")
                block_size = 2048
                multispectral_data = np.empty((src.count, int(window.height), int(window.width)), dtype=src.dtypes[0])

                for b in range(src.count):
                    for r_off in range(0, int(window.height), block_size):
                        for c_off in range(0, int(window.width), block_size):
                            current_window = Window(window.col_off + c_off,
                                                    window.row_off + r_off,
                                                    min(block_size, int(window.width) - c_off),
                                                    min(block_size, int(window.height) - r_off))
                            if current_window.width > 0 and current_window.height > 0:
                                block_data = src.read(b + 1, window=current_window)
                                multispectral_data[b, r_off:r_off+block_data.shape[0], c_off:c_off+block_data.shape[1]] = block_data

            # --- 转换为全色影像的核心修改：增加归一化到uint8 ---
            if multispectral_data.shape[0] > 1: # 确保是多波段影像
                # 将所有波段数据转换为 float32 进行计算，以保持精度
                pan_data_float = np.mean(multispectral_data.astype(np.float32), axis=0)
            else:
                # 如果本身就是单波段，则直接使用
                pan_data_float = multispectral_data[0].astype(np.float32)

            # 归一化到 0-255 范围
            min_val = np.min(pan_data_float)
            max_val = np.max(pan_data_float)

            if max_val > min_val: # 避免除以零
                # 线性拉伸到 0-255，并转换为 uint8
                pan_data_uint8 = ((pan_data_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else: # 如果所有像素值都相同 (平坦区域)
                pan_data_uint8 = np.zeros_like(pan_data_float, dtype=np.uint8)
                if min_val > 0: # 如果值不是0，则设为最大值255
                    pan_data_uint8.fill(255)


            # 确保 pan_data_uint8 是一个二维数组 (高度, 宽度)
            if pan_data_uint8.ndim == 3 and pan_data_uint8.shape[0] == 1:
                pan_data_uint8 = pan_data_uint8.squeeze()
            elif pan_data_uint8.ndim != 2:
                raise ValueError("转换后的全色数据维度不正确，预期为二维。")

            # 创建新的变换矩阵
            out_transform = src.window_transform(window)

            output_filename = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_pan_uint8.tif") # 修改文件名以区分

            # 写入输出影像，现在是单波段，uint8类型
            with rasterio.open(
                output_filename,
                'w',
                driver='GTiff',
                height=pan_data_uint8.shape[0],
                width=pan_data_uint8.shape[1],
                count=1,
                dtype=np.uint8,  # 明确指定为 uint8
                crs=src.crs,
                transform=out_transform,
                nodata=None      # 由于归一化，通常不再保留原始的nodata值，除非你希望对nodata像素进行特殊处理
                                 # 如果原始影像有nodata，并且你希望它们在uint8输出中仍然是nodata，需要更复杂的处理
                                 # 例如，在归一化之前屏蔽nodata像素，并在归一化后再将其设回。
            ) as dst:
                dst.write(pan_data_uint8, 1) # 写入第一个波段
        print(f"已生成全色交集影像 (uint8): {output_filename}")



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