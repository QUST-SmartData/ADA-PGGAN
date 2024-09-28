# from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from interfacegan.models.pggan_generator import PGGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import ImageToLatent, ImageLatentDataset
from models.losses import LogCoshLoss
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from glob import glob
from tqdm import tqdm_notebook as tqdm
import numpy as np
from PIL import Image
import os

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

'''Create Dataloaders
Using a 50,000 image dataset. Generated with the generated_data.py script at https://github.com/ShenYujun/InterFaceGAN.'''


augments = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485],
#                          std=[0.229])
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_size = 1024

directory = "./gen_train100k/"
filenames = sorted(glob(directory + "*.png"))
# 定义存储转换后图像文件名的列表
# converted_filenames = []

# # 循环遍历所有文件名
# for filename in filenames:
#     # 打开图像文件
#     image = Image.open(filename)
#     # 检查图像的通道数
#     if image.mode != 'RGB':
#         # 如果不是三通道RGB图像，则将其转换为RGB格式
#         image = image.convert("RGB")
#         # 生成新的文件名，添加到converted_filenames列表中
#         new_filename = os.path.splitext(filename)[0] + "_rgb.jpg"
#         image.save(new_filename)  # 保存转换后的图像
#         converted_filenames.append(new_filename)
#     else:
#         converted_filenames.append(filename)
#     # 关闭图像文件
#     image.close()


# # 定义存储转换后图像文件名的列表
# converted_filenames = []

# # 循环遍历所有文件名
# for idx, filename in enumerate(filenames, start=1):
#     # 打开图像文件
#     image = Image.open(filename)
#     # 检查图像的通道数
#     if image.mode != 'RGB':
#         # 如果不是三通道RGB图像，则将其转换为RGB格式
#         image = image.convert("RGB")
#         # 生成新的文件名，添加到converted_filenames列表中
#         new_filename = os.path.splitext(filename)[0] + "_rgb.jpg"
#         image.save(new_filename)  # 保存转换后的图像
#         converted_filenames.append(new_filename)
#     else:
#         converted_filenames.append(filename)
#     # 关闭图像文件
#     image.close()

train_filenames = filenames[0:98000]
validation_filenames = filenames[98000:]

dlatents = np.load(directory + "wp.npy")

train_dlatents = dlatents[0:98000]
validation_dlatents = dlatents[98000:]

train_dataset = ImageLatentDataset(train_filenames, train_dlatents, transforms=augments)
validation_dataset = ImageLatentDataset(validation_filenames, validation_dlatents, transforms=augments)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=8)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=8)


'Instantiate Model'
image_to_latent = ImageToLatent(image_size).cuda()
optimizer = torch.optim.Adam(image_to_latent.parameters())
criterion = LogCoshLoss()

'Train Model'
epochs = 20
validation_loss = 0.0

progress_bar = tqdm(range(epochs))

device = torch.device('cuda')
with dnnlib.util.open_url('./network-snapshot-000600.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
c = None    

for epoch in progress_bar:    
    
    running_loss = 0.0
    
    image_to_latent.train()
    print('开始训练++++++++++++++++++++++++')
    for i, (images, latents) in enumerate(train_generator,1):
#         print('i+++++++++++++++epoch',{i}_{epoch})
        print(f'i+++++++++++++++epoch {i}_{epoch}')  # 使用 f-string 进行字符串格式化
#         print('images.size(1)++++++++++++++++++++++++++++++',images.size(1))
        optimizer.zero_grad()
#         print('images.size(1)++++++++++++++++++++++++++',images.size(1))
        if images.size(1) == 1:  # 如果图像通道数为1，表示单通道灰度图像
            # 如果是单通道图像，复制通道以使其变为3通道（假设是灰度图像）
            images = torch.cat([images, images, images], dim=1)
#             print('完成更换++++++++++++++++++++++')

        images, latents = images.cuda(), latents.cuda()
        pred_latents = image_to_latent(images)
        
        pred_latents = G.mapping(pred_latents, c, truncation_psi=0.5, truncation_cutoff=8)
        
        print('pred_latents',pred_latents.shape)
#         print('pred_latents',pred_latents)
        print('latents',latents.shape)
        loss = criterion(pred_latents, latents)
        print('loss============================',loss)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))
    
    validation_loss = 0.0
    
    image_to_latent.eval()
    for i, (images, latents) in enumerate(validation_generator,1):
        print('val_i===============',i)
        with torch.no_grad():
            images, latents = images.cuda(), latents.cuda()
            pred_latents = image_to_latent(images)
            print('latents',latents.shape)
            pred_latents = G.mapping(pred_latents, c, truncation_psi=0.5, truncation_cutoff=8)
            loss =  criterion(pred_latents, latents)
            
            validation_loss += loss.item()
    
    validation_loss /= i
    progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))
    
    
    
# dllaten_test=image_to_latent(test)



def normalized_to_normal_image(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).float()
    std=torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).float()
    
    image = image.detach().cpu()
    
    image *= std
    image += mean
    image *= 255
    
    image = image.numpy()[0]
    image = np.transpose(image, (1,2,0))
    return image.astype(np.uint8)


num_test_images = 5
images = [validation_dataset[i][0].unsqueeze(0).cuda() for i in range(num_test_images)]
normal_images = list(map(normalized_to_normal_image, images))

pred_dlatents = map(image_to_latent, images)
print('dllaten_test',pred_dlatents)
#Save Model   
torch.save(image_to_latent.state_dict(), "./image_to_latent_styleada_w10k.pt")

#Load Model
# image_to_latent = ImageToLatent(image_size).cuda()
# image_to_latent.load_state_dict(torch.load("image_to_latent.pt"))
# image_to_latent.eval()
