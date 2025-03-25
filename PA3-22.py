import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import matplotlib.font_manager as fm
import os

# ======================
# 配置模块
# ======================
class OcrConfig:
    def __init__(self):
        # 字体配置
        self.font_path = 'E:\source-han-sans-release\OTF\SimplifiedChinese\SourceHanSansSC-Normal.otf'
        self._verify_font()  # 正确调用类方法

        # OCR引擎配置
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=False,
            show_log=False
        )
    def _verify_font(self):  # 正确缩进，与__init__同级
        """验证中文字体文件是否存在"""
        if not os.path.exists(self.font_path):
            raise FileNotFoundError(f"字体文件 {self.font_path} 未找到")
        font_prop = fm.FontProperties(fname=self.font_path)
        plt.rcParams['font.family'] = font_prop.get_name()

# ======================
# 图像处理模块
# ======================
class ImageProcessor:
    @staticmethod
    def preprocess(img_path):
        """图像预处理流水线"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像文件: {img_path}")
            
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(
            gray, 
            h=15, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

# ======================
# OCR处理模块
# ======================
class PaddleOcrEngine:
    def __init__(self, config):
        self.ocr = config.ocr_engine
        
    def extract_text(self, processed_img):
        """执行OCR识别"""
        # 转换为BGR格式（PaddleOCR输入要求）
        img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        # 执行OCR
        result = self.ocr.ocr(img_bgr, cls=True)
        
        # 解析结果
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(f"{text} ({confidence:.2f})")
        return '\n'.join(texts) if texts else "未检测到文本"

# ======================
# 可视化模块
# ======================
class ResultVisualizer:
    @staticmethod
    def visualize(original_img, processed_img, ocr_result):
        """可视化对比结果"""
        plt.figure(figsize=(16, 12))
        
        # 原始图像
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('原始图像', fontsize=12)
        plt.axis('off')
        
        # 预处理图像
        plt.subplot(2, 2, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title('预处理结果', fontsize=12)
        plt.axis('off')
        
        # 文本结果
        plt.subplot(2, 1, 2)
        plt.text(0.5, 0.5, ocr_result, 
                 ha='center', va='center', 
                 fontsize=10, 
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# ======================
# 主流程
# ======================
if __name__ == "__main__":
    try:
        # 初始化配置
        config = OcrConfig()
        
        # 图像处理
        processor = ImageProcessor()
        original_img = cv2.imread('E:/opencv/3-22.png')
        processed_img = processor.preprocess('E:/opencv/3-22.png')
        
        # OCR识别
        ocr_engine = PaddleOcrEngine(config)
        result_text = ocr_engine.extract_text(processed_img)
        
        # 结果展示
        ResultVisualizer.visualize(original_img, processed_img, result_text)
        
    except Exception as e:
        print(f"程序异常: {str(e)}")