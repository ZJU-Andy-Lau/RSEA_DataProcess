import cv2
import numpy as np
import os
import argparse

def calculate_cloud_percentage(image_path):
    """
    计算单波段遥感图像的云量。

    Args:
        image_path (str): 图像文件的路径。

    Returns:
        tuple: (云量百分比, 处理后的图像) 如果成功，否则 (None, None)。
    """
    try:
        # 1. 读取图像
        # OpenCV 默认以BGR格式读取彩色图像，对于单波段图像，它会读取为灰度图
        # 如果您的PNG图像是单通道的，imread会直接读取为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"错误：无法读取图像文件 {image_path}。请检查路径和文件格式。")
            return None, None

        # 2. (可选) 预处理：直方图均衡化
        # 适用于增强图像对比度，尤其是在光照不均的情况下
        # 您可以根据实际图像效果决定是否启用
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced_image = clahe.apply(image)
        enhanced_image = image # 暂时不进行预处理，直接使用原始图像

        # 3. 阈值分割：使用 Otsu's 方法自动确定阈值
        # Otsu's 方法返回最佳阈值和二值化后的图像
        # cv2.THRESH_BINARY 表示大于阈值的像素设置为最大值（255），否则设置为0
        # cv2.THRESH_OTSU 表示使用Otsu's方法自动寻找阈值
        ret, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"图像 {os.path.basename(image_path)} 的 Otsu 阈值为: {ret}")

        # 4. (可选) 形态学操作：去除噪声和填充小空洞
        # 定义结构元素 (通常是矩形或椭圆形)
        kernel = np.ones((5, 5), np.uint8) # 5x5 的矩形结构元素

        # 开运算 (Opening): 先腐蚀后膨胀，用于去除小的白色噪声点
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

        # 闭运算 (Closing): 先膨胀后腐蚀，用于填充小的黑色空洞
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5. 云量计算
        # 统计二值图像中白色像素 (云) 的数量
        cloud_pixels = np.sum(closed_image == 255)
        total_pixels = closed_image.size
        cloud_cover_percentage = (cloud_pixels / total_pixels) * 100

        return cloud_cover_percentage

    except Exception as e:
        print(f"处理图像 {image_path} 时发生错误: {e}")
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()

    folders = os.listdir(options.root)
    folders = sorted(folders,key=int)

    for folder in folders:
        cp1 = calculate_cloud_percentage(os.path.join(options.root,folder,'image_0.png'))
        cp2 = calculate_cloud_percentage(os.path.join(options.root,folder,'image_1.png'))
        print(folder,max(cp1,cp2))
    
