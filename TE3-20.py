import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

#配置Matplotlib使用中文字体
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 

# 全局设置字体
font_path = 'E:\source-han-sans-release\OTF\SimplifiedChinese\SourceHanSansSC-Normal.otf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


# 配置Tesseract路径（Windows需要，Mac不需要）
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract\tesseract.exe'

# 图像预处理
def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    return denoised

# 测试图像
image_path = 'E:/opencv/3-22.png'  # 用手机拍一张包含文字的图片

# 识别流程
processed_img = preprocess(image_path)
custom_config = r'--oem 3 --psm 6 -l eng'
text = pytesseract.image_to_string(processed_img, config=custom_config)

# 输出结果
print("识别结果：\n", text)

# 创建画布
fig = plt.figure(figsize=(12, 8))

# 添加图像区域
ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.4])  # 原始图像区域
ax2 = fig.add_axes([0.5, 0.55, 0.4, 0.4])   # 预处理图像区域

# 显示原始图像
ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
ax1.set_title('原始图像', fontproperties=font_prop, fontsize=14)
ax1.axis('off')  # 隐藏坐标轴

# 显示预处理后的图像
ax2.imshow(processed_img, cmap='gray')
ax2.set_title('预处理后', fontproperties=font_prop, fontsize=14)
ax2.axis('off')  # 隐藏坐标轴

# 添加文本框
text_box = fig.add_axes([0.1, 0.1, 0.8, 0.3])  # 文本框区域
text_box.axis('off')  # 隐藏坐标轴

# 计算文本框最大高度
max_text_box_height = 0.3  # 文本框最大高度
text_fontsize = 14  # 文本框字体大小
max_text_length = 500  # 最大文本长度

# 动态调整文本框内容
if len(text) > max_text_length:
    text_box.text(0.5, 0.5, "警告：文本内容过多", 
                  ha="center", va="center", fontsize=text_fontsize, fontproperties=font_prop,
                  color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
else:
    # 计算文本行数和所需高度
    text_lines = text.count('\n') + 1
    line_height = 0.05  # 每行文本的高度
    required_height = line_height * text_lines

    # 调整文本框高度
    if required_height > max_text_box_height:
        text_box.text(0.5, 0.5, "警告：文本内容过多", 
                      ha="center", va="center", fontsize=text_fontsize, fontproperties=font_prop,
                      color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
    else:
        text_box.text(0.5, 0.5, f"识别结果：\n{text}", 
                      ha="center", va="center", fontsize=text_fontsize, fontproperties=font_prop,
                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# 调整布局
plt.tight_layout()
plt.show()


