


import numpy as np
import dnnlib
import legacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            UnFlatten(),
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            self._block(features_g * 2, features_g * 2, 4, 2, 1),  # img: 32x32
            # self._block(features_g * 2, features_g * 2, 4, 2, 1),  # img: 32x32

            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    
    
LEARNING_RATE = 1e-4
# BATCH_SIZE = 40
IMAGE_SIZE = 128
CHANNELS_IMG = 1
Z_DIM = 1024
NUM_EPOCHS = 10
FEATURES_CRITIC = 16
FEATURES_GEN = 32
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
# load model
# latent_dim = 1024
# generator = Generator(latent_dim)
# generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
# generator.load_state_dict(torch.load('./generator_model.pth'))  # Assuming the model is saved in PyTorch format
# with dnnlib.util.open_url('./network-snapshot-000600.pkl', 'rb') as f:
# #     generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
#     generator = legacy.load_network_pkl(f)['G_ema'] # type: ignore
# c = None       
# def generate_random_latent_vectors(num_vectors, latent_dim):
#     return np.random.randn(num_vectors, latent_dim)
device = torch.device('cuda')
with dnnlib.util.open_url('./network-snapshot-000600.pkl', 'rb') as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
c = None   
# c = c.to(device)

def generate_random_latent_vectors(latent_dim, num_vectors):
    return torch.randn(num_vectors, latent_dim)

# def interpolate_vectors(vec1, vec2, num_interpolations):
#     ratios = np.linspace(0, 1, num_interpolations)[:, np.newaxis]
#     return ratios * vec1 + (1 - ratios) * vec2


# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = torch.acos(torch.clamp(torch.dot(low / torch.norm(low), high / torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0 - val) * low + val * high
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high


# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=12):
    ratios = torch.linspace(0, 1, steps=n_steps)
    vectors = []
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return torch.stack(vectors)




# 定义潜在向量的维度和插值数量
latent_dim = 512
num_vectors = 10
num_interpolations = 12

# 生成随机潜在向量
random_latent_vectors = generate_random_latent_vectors(latent_dim,num_vectors)
print('random_latent_vectors',random_latent_vectors.shape)

i=1
for latent_vector in random_latent_vectors:
#     prinr('j',j)
    print('latent_vector ',latent_vector.shape)
    latent_vector = latent_vector.to('cuda')

    latent_vector = torch.tensor(latent_vector,dtype=torch.float32)
#     print('latent_vector ',latent_vector.shape)
    latent_vector = latent_vector.view(1, 512)
    # 将潜在向量输入到生成器中生成图像
    generated_image = generator(latent_vector,c,noise_mode='const')
    generated_images = generated_image.view(-1, 1, 1024, 1024)
    save_image(generated_images, f"./result2/rock_{i}.png",  normalize=True)
    i+=1



# 对随机潜在向量进行插值
interpolated_vectors = []

# 保存最初生成的十个潜在向量
interpolated_vectors.append(random_latent_vectors[0])
print('random_latent_vectors[0]',random_latent_vectors[0].shape)

for i in range(num_vectors - 1):
    interpolated_vectors.extend(interpolate_points(random_latent_vectors[i], random_latent_vectors[i + 1]))
    # 在每个插值之后保存下一个原始潜在向量
    interpolated_vectors.append(random_latent_vectors[i + 1])

# interpolated_vectors = np.array(interpolated_vectors)
# print('interpolated_vectors', interpolated_vectors.shape)
j=1
for latent_vector in interpolated_vectors:
#     prinr('j',j)
    print('latent_vector ',latent_vector.shape)
    latent_vector = latent_vector.to('cuda')

    latent_vector = torch.tensor(latent_vector,dtype=torch.float32)
    print('latent_vector ',latent_vector.shape)
    latent_vector = latent_vector.view(1, 512)
    # 将潜在向量输入到生成器中生成图像
    generated_image = generator(latent_vector,c,noise_mode='const')
    generated_images = generated_image.view(-1, 1, 1024, 1024)
    save_image(generated_images, f"./result1/rock_{j}.png",  normalize=True)
    j+=1
