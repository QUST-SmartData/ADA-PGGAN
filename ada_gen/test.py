import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from .models.progressive_gan import ProgressiveGAN
from models.progressive_gan import ProgressiveGAN


generator = ProgressiveGAN()

# 定义输出目录和标签
outDir = "./network1/output_networks/celebaHQ"
outLabel = "celebaHQ_s8_i96000"
# 生成保存模型的路径
pathModel = os.path.join(outDir, outLabel + ".pt")
obj = torch.load(pathModel)
print(type(obj))  # 应该输出 <class 'torch.nn.Module'> 或者你定义的模型类
torch.save(obj, './path_to_save_model.pt')
loaded_model = torch.load('path_to_save_model.pt')
print(type(loaded_model))

import torch

# 加载文件
obj = torch.load(pathModel)

# 检查obj的类型
if isinstance(obj, torch.nn.Module):
    print("整个模型对象已被保存。")
elif isinstance(obj, dict):
    print("模型的状态字典已被保存。")
else:
    print("保存的内容是未知类型。")










torch.save(generator, pathModel)
generator.load_state_dict(torch.load(pathModel))
generator.eval()


# 保存整个模型
# torch.save(model, pathModel)

# # 加载整个模型
# loaded_model = torch.load(pathModel)

# # 将模型设为评估模式
# loaded_model.eval()

# 定义一个函数生成图像
# def generate_image(model, input_vector):
#     with torch.no_grad():  # 禁用梯度计算
#         generated_image = model(input_vector).view(1024, 1024)  # 假设输出是28x28图像
#     return generated_image

# 生成随机噪声向量，假设输入维度为100
random_input = torch.randn(1, 512)

# 生成图像
generated_image = generator(random_input)

# 显示生成的图像
plt.imshow(generated_image.numpy(), cmap='gray')
plt.savefig('test1.png')
plt.show()
