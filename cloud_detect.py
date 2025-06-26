import cv2
import numpy as np
import os
import argparse

def calculate_cloud_percentage(image_path):
    """
    计算单波段遥感图像中的云量百分比。
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    # 1. 加载图像 (以灰度模式加载，因为是单波段)
    # cv2.imread的第二个参数0表示以灰度模式读取图像
    image = cv2.imread(image_path, 0)

    if image is None:
        print(f"Error: Could not open or find the image at {image_path}. "
              "Please ensure it's a valid image file.")
        return None

    # 2. 灰度分析 (可选，用于调试和理解图像特性)
    # print(f"Image min pixel value: {np.min(image)}")
    # print(f"Image max pixel value: {np.max(image)}")
    # print(f"Image mean pixel value: {np.mean(image)}")

    # 3. 阈值选择 - 使用 Otsu's 方法
    # retVal是计算出的阈值，thresh_image是二值化后的图像
    # cv2.THRESH_BINARY: 大于阈值的像素设为 maxval，否则设为 0
    # cv2.THRESH_OTSU: 使用 Otsu's 算法自动确定最佳阈值
    retVal, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f"Otsu's threshold value: {retVal}") # 打印Otsu's算法确定的阈值

    # 4. 云像素计数
    # 统计二值化图像中白色像素（云）的数量
    # thresh_image == 255 会创建一个布尔数组，其中云像素为 True，非云像素为 False
    # np.sum() 会将 True 视为 1，False 视为 0，从而统计出云像素的数量
    cloud_pixels = np.sum(thresh_image == 255)

    # 5. 云量计算
    total_pixels = image.shape[0] * image.shape[1] # 图像高度 * 图像宽度
    cloud_percentage = (cloud_pixels / total_pixels) * 100

    return cloud_percentage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    folders = os.listdir(options.root)

    for folder in folders:
        cp1 = calculate_cloud_percentage(os.path.join(options.root,folder,'image_0.png'))
        cp2 = calculate_cloud_percentage(os.path.join(options.root,folder,'image_1.png'))
        print(folder,max(cp1 * 100,cp2 * 100))