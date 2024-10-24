import torch.optim as optim
import torchvision.transforms as transforms
import random

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.progressive_conv_net import GNet, DNet


class ProgressiveGAN(BaseGAN):
    r"""
    Implementation of NVIDIA's progressive GAN with Adaptive Data Augmentation (ADA).
    """

    def __init__(self,
                 dimLatentVector=512,
                 depthScale0=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 ada_enabled=True,  # 新增的参数：是否启用自适应数据增强
                 ada_p=0.6,         # 数据增强的初始概率
                 **kwargs):
        r"""
        Args:

        Specific Arguments:
            - ada_enabled (bool): 是否启用自适应数据增强
            - ada_p (float): 数据增强的初始概率，影响翻转等操作的概率

        其他参数与之前相同...
        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.depthScale0 = depthScale0
        self.config.initBiasToZero = initBiasToZero
        self.config.leakyReluLeak = leakyness
        self.config.depthOtherScales = []
        self.config.perChannelNormalization = perChannelNormalization
        self.config.alpha = 0
        self.config.miniBatchStdDev = miniBatchStdDev
        self.config.equalizedlR = equalizedlR
        self.ada_enabled = ada_enabled
        self.ada_p = ada_p  # 自适应数据增强的概率

        BaseGAN.__init__(self, dimLatentVector, **kwargs)

        # 定义增强变换
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=self.ada_p),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=self.ada_p)  # 随机垂直翻转
        ])

    def getNetG(self):
        gnet = GNet(self.config.latentVectorDim,
                    self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    normalization=self.config.perChannelNormalization,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # 添加新的尺度
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # 如果有新尺度，给生成器添加混合层
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):
        dnet = DNet(self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    self.config.categoryVectorDim,
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # 添加新的尺度
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # 如果有新尺度，给判别器添加混合层
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return dnet

    def apply_data_augmentation(self, images):
        r"""
        根据 self.ada_p 概率对输入图像应用数据增强。
        """
        if self.ada_enabled:
            augmented_images = self.augment(images)
            return augmented_images
        return images

    def adjust_ada_probability(self, d_loss):
        r"""
        动态调整数据增强概率 ada_p，根据判别器的损失变化进行自适应调整。
        """
        if self.ada_enabled:
            # 假设目标是判别器损失接近某个阈值
            target_loss_threshold = self.config.target_loss_threshold if hasattr(self.config, 'target_loss_threshold') else 0.6
            if d_loss > target_loss_threshold:
                # 判别器表现较差，增大数据增强概率
                self.ada_p = min(self.ada_p + 0.05, 1.0)
            else:
                # 判别器表现较好，减小数据增强概率
                self.ada_p = max(self.ada_p - 0.05, 0.0)

            # 更新增强的概率
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.ada_p),
                transforms.RandomVerticalFlip(p=self.ada_p)
            ])
            print(f"自适应数据增强的概率调整为: {self.ada_p}")

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def addScale(self, depthNewScale):
        r"""
        添加新的尺度，分辨率增大一倍。
        """
        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        self.netG.addScale(depthNewScale)
        self.netD.addScale(depthNewScale)

        self.config.depthOtherScales.append(depthNewScale)

        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        r"""
        更新混合因子 alpha。
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.getOriginalG().setNewAlpha(newAlpha)
        self.getOriginalD().setNewAlpha(newAlpha)

        if self.avgG:
            self.avgG.module.setNewAlpha(newAlpha)

        self.config.alpha = newAlpha

    def getSize(self):
        r"""
        获取输出图像的大小 (W, H)。
        """
        return self.getOriginalG().getOutputSize()

    def train_step(self, real_images, latent_vectors):
        r"""
        定义一个训练步长，其中包括生成器、判别器的前向和后向传播。
        """
        # 生成器生成假图像
        fake_images = self.netG(latent_vectors)
        
        # 对真实图像应用数据增强
        real_images = self.apply_data_augmentation(real_images)

        # 判别器对真实图像的输出
        real_output = self.netD(real_images)

        # 判别器对生成图像的输出
        fake_output = self.netD(fake_images)

        # 判别器损失
        d_loss = self.lossCriterion.calculate_discriminator_loss(real_output, fake_output)
        self.optimizerD.zero_grad()
        d_loss.backward()
        self.optimizerD.step()

        # 生成器损失和优化步骤
        g_loss = self.lossCriterion.calculate_generator_loss(fake_output)
        self.optimizerG.zero_grad()
        g_loss.backward()
        self.optimizerG.step()

        return d_loss, g_loss

    def train(self, epochs, data_loader):
        r"""
        定义完整的训练循环，其中包含自适应数据增强的更新。
        """
        for epoch in range(epochs):
            for real_images, latent_vectors in data_loader:
                d_loss, g_loss = self.train_step(real_images, latent_vectors)
                
                # 动态调整 ADA 概率
                self.adjust_ada_probability(d_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

