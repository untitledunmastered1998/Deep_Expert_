# import torch
#
#
# x_list = []
# for i in range(4):
#     x = torch.randn(32, 512)
#     x_list.append(x)
# x = torch.stack(x_list)
# x_mean = torch.stack(x_list).mean(dim=0)
# print(x.shape)
# print(x_mean.shape)

import requests
import numpy as np
import os
from tqdm import tqdm


def download_laion_features(output_dir="/data/datasets/laion400M"):
    """
    分片下载LAION-400M特征
    """
    os.makedirs(output_dir, exist_ok=True)

    # 设置下载参数
    chunk_size = 1000000  # 每个分片包含100万个特征
    total_images = 400000000

    for start_idx in tqdm(range(0, total_images, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_images)

        # 构建分片文件名
        output_file = f"{output_dir}/features_{start_idx}_{end_idx}.npy"

        if os.path.exists(output_file):
            print(f"分片 {output_file} 已存在，跳过")
            continue

        try:
            # 下载特征分片
            response = requests.get(f"https://api.laion.ai/features/{start_idx}/{end_idx}")
            features = np.load(response.content)

            # 保存到本地
            np.save(output_file, features)

        except Exception as e:
            print(f"下载分片 {start_idx}-{end_idx} 时出错: {str(e)}")

download_laion_features()