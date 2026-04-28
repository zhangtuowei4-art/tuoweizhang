import os
import numpy as np
from PIL import Image

# ==========================
# 1. 数据加载模块 (优化内存与速度)
# ==========================
class EuroSATLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 获取类别名并排序，确保索引一致性
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def load_data(self):
        """
        预先计算总数并使用预分配数组，比 list.append 更节省内存且速度更快
        """
        all_files = []
        for cls in self.class_names:
            cls_path = os.path.join(self.root_dir, cls)
            for f in os.listdir(cls_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_files.append((os.path.join(cls_path, f), self.class_to_idx[cls]))

        num_samples = len(all_files)
        # EuroSAT 64x64x3 展平后为 12288
        X = np.empty((num_samples, 12288), dtype=np.float32)
        y = np.empty(num_samples, dtype=np.int64)

        print(f"开始加载 {num_samples} 张图像...")
        for i, (img_path, label) in enumerate(all_files):
            img = Image.open(img_path).convert("RGB")
            # 归一化并展平
            X[i] = np.array(img, dtype=np.float32).reshape(-1) / 255.0
            y[i] = label
            if (i + 1) % 5000 == 0:
                print(f"已加载 {i + 1}/{num_samples}")

        return X, y

if __name__ == "__main__":
    # 测试数据加载
    data_dir = "EuroSAT_RGB" 
    if not os.path.exists(data_dir):
        print(f"错误: 未找到数据集目录 '{data_dir}'")
    else:
        loader = EuroSATLoader(data_dir)
        X, y = loader.load_data()
        print(f"数据加载完成，X形状: {X.shape}, y形状: {y.shape}")