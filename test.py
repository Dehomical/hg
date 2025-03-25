"""
OCR性能对比测试系统
功能:对比Tesseract与PaddleOCR在速度、精度、资源消耗方面的表现
作者:DEHO
版本1.1
日期:2025-3
"""

# ======================
# 导入依赖库
# ======================
import os
import time
import psutil
import cv2
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
from Levenshtein import distance as levenshtein_distance  # 编辑距离计算

# ======================
# 配置模块
# ======================
class Config:
    """全局配置参数"""
    def __init__(self):
        # Tesseract路径配置（需根据实际情况修改）
        self.tesseract_path = r'E:\Tesseract\tesseract.exe'
        # PaddleOCR参数
        self.paddle_config = {
            'use_angle_cls': True,   # 启用方向分类
            'lang': 'ch',            # 中文识别
            'use_gpu': False,         # 使用CPU模式
            'show_log': False         # 关闭日志输出
        }
        # 测试参数
        self.test_params = {
            'warmup_runs': 3,        # 预热次数
            'test_runs': 10,          # 正式测试次数
            'min_confidence': 0.6     # 置信度阈值
        }
        # 初始化环境
        self._init_env()

    def _init_env(self):
        """环境初始化"""
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False

# ======================
# 预处理模块
# ======================
class TextPreprocessor:
    """文本预处理工具"""
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        统一文本格式以提高评估准确性
        :param text: 原始文本
        :return: 标准化后的文本
        """
        # 转换全角字符为半角
        text = text.translate(str.maketrans('　Ａ-Ｚａ-ｚ０-９', ' A-Za-z0-9'))
        # 移除所有空格和换行符
        text = text.replace(' ', '').replace('\n', '')
        # 统一常用符号
        replacements = {'：': ':', '；': ';', '，': ',', '。': '.', '‘': "'"}
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.lower()  # 统一为小写

# ======================
# OCR引擎模块
# ======================
class TesseractEngine:
    """Tesseract OCR处理器"""
    def process(self, img_path: str) -> str:
        """Tesseract处理流程"""
        img = Image.open(img_path)
        return pytesseract.image_to_string(
            img, 
            lang='chi_sim+eng', 
            config='--psm 6 --oem 3'
        )

class PaddleOCREngine:
    """PaddleOCR处理器"""
    def __init__(self, config: dict):
        self.ocr = PaddleOCR(**config)

    def process(self, img_path: str) -> str:
        """PaddleOCR处理流程"""
        result = self.ocr.ocr(img_path, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2 and line[1][1] >= Config().test_params['min_confidence']:
                    texts.append(line[1][0])
        return '\n'.join(texts)

# ======================
# 性能测试模块
# ======================
class OCRBenchmarker:
    """OCR性能对比测试器"""
    def __init__(self, test_data_dir: str, gt_dir: str):
        """
        :param test_data_dir: 测试图像目录
        :param gt_dir: 标注文本目录
        """
        # 验证目录有效性
        if not os.path.isdir(test_data_dir):
            raise ValueError(f"无效测试目录: {test_data_dir}")
        if not os.path.isdir(gt_dir):
            raise ValueError(f"无效标注目录: {gt_dir}")

        # 加载数据集
        self.test_images = self._load_images(test_data_dir)
        self.gt_texts = self._load_gt_texts(gt_dir)

    def _load_images(self, dir_path: str) -> list:
        """加载测试图像路径"""
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']
        return sorted([
            os.path.join(dir_path, f) 
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])

    def _load_gt_texts(self, dir_path: str) -> list:
        """加载标注文本"""
        return sorted([
            open(os.path.join(dir_path, f), 'r', encoding='utf-8').read().strip()
            for f in os.listdir(dir_path)
            if f.endswith('.txt')
        ])

    def _calculate_similarity(self, truth: str, pred: str) -> float:
        """
        基于编辑距离计算文本相似度
        :return: 相似度分数 (0.0~1.0)
        """
        # 预处理文本
        truth_clean = TextPreprocessor.preprocess_text(truth)
        pred_clean = TextPreprocessor.preprocess_text(pred)

        # 处理空文本特殊情况
        if len(truth_clean) == 0 and len(pred_clean) == 0:
            return 1.0
        if len(truth_clean) == 0 or len(pred_clean) == 0:
            return 0.0

        # 计算相似度
        edit_dist = levenshtein_distance(truth_clean, pred_clean)
        max_len = max(len(truth_clean), len(pred_clean))
        return 1 - (edit_dist / max_len)

    def benchmark(self, ocr_func, engine_name: str) -> dict:
        """执行性能测试"""
        metrics = {
            'time': [],
            'similarity': [],
            'memory': []
        }

        print(f"\n正在测试 {engine_name} ...")

        # 预热阶段
        for _ in range(Config().test_params['warmup_runs']):
            _ = ocr_func(self.test_images[0])

        # 正式测试
        for img_path, truth in zip(self.test_images, self.gt_texts):
            try:
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024**2  # MB

                # 执行OCR
                start_time = time.time()
                pred = ocr_func(img_path)
                duration = time.time() - start_time

                # 计算指标
                mem_after = process.memory_info().rss / 1024**2
                similarity = self._calculate_similarity(truth, pred)

                metrics['time'].append(duration)
                metrics['similarity'].append(similarity)
                metrics['memory'].append(mem_after - mem_before)

                # 调试输出（可选）
                print(f"\n[样本] {os.path.basename(img_path)}")
                print(f"标注文本: {truth[:50]}...")  # 显示前50字符
                print(f"识别结果: {pred[:50]}...")
                print(f"相似度: {similarity:.2%}")

            except Exception as e:
                print(f"处理 {os.path.basename(img_path)} 失败: {str(e)}")

        return {k: np.mean(v) for k, v in metrics.items()}

    def visualize_report(self, tesseract_metrics: dict, paddle_metrics: dict):
        """生成可视化对比报告"""
        labels = ['时间 (秒)', '相似度', '内存增量 (MB)']
        t_values = [
            tesseract_metrics['time'],
            tesseract_metrics['similarity'],
            tesseract_metrics['memory']
        ]
        p_values = [
            paddle_metrics['time'],
            paddle_metrics['similarity'],
            paddle_metrics['memory']
        ]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, t_values, width, label='Tesseract')
        rects2 = ax.bar(x + width/2, p_values, width, label='PaddleOCR')

        ax.set_ylabel('性能指标')
        ax.set_title('OCR引擎性能对比报告')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig('performance_report.png', dpi=300)
        plt.show()

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    try:
        # 初始化配置
        config = Config()

        # 设置测试路径（根据实际情况修改）
        TEST_IMAGES_DIR = "E:/opencv/test_images"
        GROUND_TRUTH_DIR = "E:/opencv/ground_truth"
        # 初始化测试器
        benchmarker = OCRBenchmarker(TEST_IMAGES_DIR, GROUND_TRUTH_DIR)

        # 测试Tesseract
        tesseract_engine = TesseractEngine()
        tesseract_metrics = benchmarker.benchmark(
            tesseract_engine.process, 
            "Tesseract"
        )

        # 测试PaddleOCR
        paddle_engine = PaddleOCREngine(config.paddle_config)
        paddle_metrics = benchmarker.benchmark(
            paddle_engine.process,
            "PaddleOCR"
        )

        # 打印结果
        print("\n【测试结果汇总】")
        print("Tesseract 指标:")
        print(f"平均耗时: {tesseract_metrics['time']:.2f}s")
        print(f"平均相似度: {tesseract_metrics['similarity']:.2%}")
        print(f"内存增量: {tesseract_metrics['memory']:.1f}MB")

        print("\nPaddleOCR 指标:")
        print(f"平均耗时: {paddle_metrics['time']:.2f}s")
        print(f"平均相似度: {paddle_metrics['similarity']:.2%}")
        print(f"内存增量: {paddle_metrics['memory']:.1f}MB")

        # 生成可视化报告
        benchmarker.visualize_report(tesseract_metrics, paddle_metrics)
        print("\n性能报告已保存为 performance_report.png")

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        print("请检查：")
        print("1. 测试目录结构是否正确")
        print("2. Tesseract路径配置")
        print("3. 依赖库是否安装")