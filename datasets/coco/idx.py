import numpy as np

# 你的 .npy 文件路径
npy_path = "datasets/coco/cls_labels_onehot.npy"

# 加载 npy 文件
data = np.load(npy_path, allow_pickle=True).item()

# 遍历每个索引并打印形状
for key, value in data.items():
    print(f"{key}: {value.shape}")