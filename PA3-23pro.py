import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# ======================
# 超分辨率模块
# ======================
class SuperResolution:
    def __init__(self, model_path='E:\Microsoft VS Code\project\project3.20\EDSR_x4.pb', scale=4):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel('edsr', scale)
    
    def enhance(self, img_path):
        """输入图像路径，返回超分后图像"""
        return self.sr.upsample(cv2.imread(img_path))

# ======================
# OCR处理模块
# ======================
class PaddleOCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
    
    def process(self, img):
        """输入OpenCV图像，返回识别文本"""
        result = self.ocr.ocr(img, cls=True)
        # 修复语法错误：补充闭合括号
        return '\n'.join([line[1][0] for line in result[0]] if result else [])

# ======================
# 一体化处理流程
# ======================
def ocr_pipeline(img_path, use_sr=True):
    # 超分辨率增强
    sr_img = SuperResolution().enhance(img_path) if use_sr else cv2.imread(img_path)
    
    # 预处理
    gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15, 
                                      templateWindowSize=7, 
                                      searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(denoised)
    
    # OCR识别
    text = PaddleOCRProcessor().process(processed)
    
    return sr_img, processed, text

# ======================
# 可视化展示
# ======================
def show_results(original, processed, text):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Processed')
    
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, text, ha='center', va='center', wrap=True)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ======================
# 执行驱动
# ======================
if __name__ == "__main__":
    img_path = 'E:/opencv/3-22.png'  # 替换为实际路径
    
    try:
        sr_img, processed_img, result = ocr_pipeline(img_path)
        show_results(sr_img, processed_img, result)
    except Exception as e:
        print(f"执行出错: {str(e)}")
        print("常见问题排查：")
        print("1. 确保EDSR_x4.pb模型文件存在")
        print("2. 检查图像路径是否正确")
        print("3. 安装所有依赖库")