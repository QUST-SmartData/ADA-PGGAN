import argparse
from tqdm import tqdm
import numpy as np
import torch
# from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from interfacegan.models.pggan_generator import PGGANGenerator
from models.latent_optimizer import LatentOptimizer
from models.image_to_latent import ImageToLatent
from models.losses import LatentLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from interfacegan.models.model_settings import MODEL_POOL

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
    # 首先，检测是否支持CUDA，并据此设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")

parser.add_argument("--save_optimized_image", default=False, help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="optimized.png", help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--video", default=False, help="Whether or not to save a video of the encoding process.", type=bool)
parser.add_argument("--video_path", default="video.avi", help="Where to save the video at.", type=str)
parser.add_argument("--save_frequency", default=10, help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="pggan_celebahq",help="The model to use from interfacegan repo.", type=str)

parser.add_argument("--learning_rate", default=0.02, help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=16, help="The VGG network layer number to extract features from.", type=int)
parser.add_argument("--use_latent_finder", default=True, help="Whether or not to use a latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--image_to_latent_path", default="image_to_latent_ada.pt", help="The path to the .pt (Pytorch) latent finder model.", type=str)

args, other = parser.parse_known_args()

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


#  """Main function."""
#   args = parse_args()
#   logger = setup_logger(args.output_dir, logger_name='generate_data')

#   logger.info(f'Initializing generator.')
#   gan_type = MODEL_POOL[args.model_name]['gan_type']
#   if gan_type == 'pggan':
#     model = PGGANGenerator(args.model_name, logger)
#     kwargs = {}

# device = torch.device('cuda')
# with dnnlib.util.open_url('./network-snapshot-000600.pkl', 'rb') as f:
#         G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
# c = None    


def optimize_latents():
    print("Optimizing Latents.")

#     synthesizer = PGGANGenerator(args.model_type)
    device = torch.device('cuda')
    with dnnlib.util.open_url('./network-snapshot-000600.pkl', 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    c = None    
#     c = c.to(device)
    latent_optimizer = LatentOptimizer(G, args.vgg_layer)
    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)
    
    if args.video or args.save_optimized_image:
        # Hook, saves an image during optimization to be used to create video.
        generated_image_hook = GeneratedImageHook(latent_optimizer.post_synthesis_processing, args.save_frequency)

    reference_image = load_images([args.image_path])
    print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
    reference_image = torch.from_numpy(reference_image).cuda()
    print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
#     print('reference_image',reference_image)
    reference_image = latent_optimizer.vgg_processing(reference_image)
    print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
#     print('reference_image_pro',reference_image)
    reference_features = latent_optimizer.vgg16(reference_image).detach()
    reference_image = reference_image.detach()
    print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
    if args.use_latent_finder:
        image_to_latent = ImageToLatent().cuda()
        image_to_latent.load_state_dict(torch.load(args.image_to_latent_path))
        image_to_latent.eval()

        print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
        
#         reference_image = normalized_to_normal_image(reference_image)
#         print('reference_image+++++++++++++++++++++++++++++++++==',reference_image)
#        
        latents_to_be_optimized = image_to_latent(reference_image)
        print('latents_to_be_optimized.shape',latents_to_be_optimized.shape)
        latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
    else:
        # 设置随机种子以便复现结果，可以移除这行以生成不同的随机数
        torch.manual_seed(0)

        # 生成一个 (1, 1024) 的标准正态分布张量
#         latents_to_be_optimized =torch.randint(0, 256, (1, 1024), dtype=torch.int).cuda().requires_grad_(True)
        latents_to_be_optimized = torch.randint(0, 256, (1, 512)).float().cuda()

# 启用梯度计算，用于优化
        latents_to_be_optimized.requires_grad_(True)
        print('latents_to_be_optimized.requires_grad',latents_to_be_optimized.requires_grad)
# 打印结果
        print('latents_to_be_optimized',latents_to_be_optimized)
#         latents_to_be_optimized = torch.zeros((1,1024)).cuda().requires_grad_(True)

    criterion = LatentLoss()
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)

    progress_bar = tqdm(range(args.iterations))
    for step in progress_bar:
        optimizer.zero_grad()
#         print('latents_to_be_optimized',latents_to_be_optimized.shape)
        generated_image_features = latent_optimizer(latents_to_be_optimized).requires_grad_(True)
#         generated_image_features= generated_image_features.requires_grad_(True)
#         print('generated_image_features',generated_image_features)
#         print('reference_features',reference_features)
                                  
        print('reference_features.requires_grad',reference_features.requires_grad)
#         print('generated_image_features',generated_image_features.requires_grad)                                   print('generated_image_features.requires_grad',generated_image_features.requires_grad)
        loss = criterion(generated_image_features, reference_features)
#         # 2. 使用损失张量传递给 torch.autograd.grad()，而不是虚拟的标量张量
#         grad_tensors = torch.autograd.grad(loss, latents_to_be_optimized, create_graph=True, retain_graph=True)

# 3. 在打印梯度时使用 .item() 方法将损失转换为标量值
#         for grad_tensor in grad_tensors:
#             print('grad=======', grad_tensor.item())
        loss.backward()
        loss = loss.item()

        optimizer.step()
        progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
    
#     optimized_dlatents = latents_to_be_optimized.detach().cuda().numpy()
    optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    print('optimized_dlatents',optimized_dlatents)
    
    np.save(args.dlatent_path, optimized_dlatents)
#     optimized_dlatents = torch.from_numpy(optimized_dlatents).to(device)
#     if c is not None:
#         c = c.to(device)

#     generated_image = G(optimized_dlatents,c)
         
#     save_image(generated_image, f"./to/FANYAN.png")
#     print('保存完成')

    if args.video:
        images_to_video(generated_image_hook.get_images(), args.video_path)
    if args.save_optimized_image:
        save_image(generated_image_hook.last_image, args.optimized_image_path)

def main():
    assert(validate_path(args.image_path, "r"))
    assert(validate_path(args.dlatent_path, "w"))
    assert(1 <= args.vgg_layer <= 16)
    if args.video: assert(validate_path(args.video_path, "w"))
    if args.save_optimized_image: assert(validate_path(args.optimized_image_path, "w"))
    if args.use_latent_finder: assert(validate_path(args.image_to_latent_path, "r"))
    
    optimize_latents()

if __name__ == "__main__":
    main()


    
    


