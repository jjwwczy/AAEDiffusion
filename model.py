import math
from tqdm.auto import tqdm

from dataclasses import dataclass
from typing import Union, Optional
import random
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import torch
from dataclasses import dataclass
from diffusers import DDPMPipeline, ImagePipelineOutput,DDIMPipeline
from typing import Optional, Union, List, Tuple
from torch import Generator
class LabeledDDIMPipeline(DDIMPipeline):
    """
    Custom pipeline for high spectral data generation with labels, using DDIM.
    Inherits from `DDIMPipeline`.
    """

    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        labels,  # Class labels
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Validate that label dimensions match batch size

        # Convert labels to the appropriate format (if necessary)
        labels = torch.tensor(labels).to(self.device)

        # Initialize noise sample for high spectral data
        num_channels = self.unet.config.in_channels
        if isinstance(self.unet.config.sample_size, int):
            data_shape = (
                batch_size,
                num_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            data_shape = (batch_size, num_channels, *self.unet.config.sample_size)

        data = randn_tensor(data_shape, generator=generator, device=self.device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Use labels as conditional input
            model_output = self.unet(data, t, class_labels=labels).sample
            data = self.scheduler.step(
                model_output, t, data, eta=eta, generator=generator
            ).prev_sample

        # Process final output
        final_output = data.cpu()

        if not return_dict:
            return (final_output,)

        return ImagePipelineOutput(images=final_output)


class LabeledDDPMPipeline(DDPMPipeline):####自定义带标签的处理管线，这部分只在inference阶段使用
    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        labels,  # 类别标签参数
        batch_size: int = 1,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 1000,
        return_dict: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # 验证标签尺寸是否与批量大小相匹配

        # 转换标签为适当的形式（如果需要）
        labels = torch.tensor(labels).to(self.device)

        # 初始化噪声样本
        # 假设高光谱数据有 `num_channels` 个频道
        num_channels = self.unet.config.in_channels
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                num_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, num_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self.device)

        # 设置时间步长
        self.scheduler.set_timesteps(num_inference_steps)
############根据给定的时间步数，从第0步开始递推出结果############
        for t in self.progress_bar(self.scheduler.timesteps):
            # 使用标签作为条件输入
            model_output = self.unet(image, t, class_labels=labels).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # 处理最终输出
        # 高光谱数据可能不需要转换为 PIL 图像或 NumPy 数组
        # 因此，这里我们直接返回 Tensor
        final_output = image.cpu()

        if not return_dict:
            return (final_output,)

        return ImagePipelineOutput(images=final_output)

    

class LabeledTimeSeriesDataset(Dataset):
    def __init__(self, Xtrain, ytrain, num_samples_per_cls: int, class_to_idx: dict):
        super().__init__()
        self.Xtrain = torch.tensor(Xtrain)
        self.ytrain = torch.tensor(ytrain)
        self.class_to_idx = class_to_idx
        self.num_samples = num_samples_per_cls

        # 选择训练数据
        train_idxs = []
        for class_id, idxs in class_to_idx.items():
            idxs_list = list(idxs)
            if len(idxs_list) >= num_samples_per_cls:
                selected_idxs = random.sample(idxs_list, num_samples_per_cls)
            else:
                selected_idxs = idxs_list
            train_idxs.extend(selected_idxs)

        self.data = self.Xtrain[train_idxs]
        self.labels = self.ytrain[train_idxs]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

    def __len__(self):
        return len(self.data)

    
class LabeledTimeSeriesDataset2D(Dataset):
    def __init__(self, Xtrain, ytrain):
        super().__init__()
        self.data = torch.tensor(Xtrain)
        self.labels = torch.tensor(ytrain)

    def __getitem__(self, idx):
        sample = self.data[idx]
      
        label = self.labels[idx]
        #########把W,H,C的维度顺序变成C,H,W以适应pytorch#####
        return sample, label

    def __len__(self):
        return len(self.data)
def compute_spectral_similarity_loss(unet_output, prototypes, labels):
    """
    计算 UNet 输出与类别原型之间的光谱相似性损失。
    :param unet_output: UNet 输出的张量，形状为 (batch_size, C, H, W)
    :param prototypes: 类别原型，形状为 (num_classes, C, H, W)
    :param labels: 每个样本的类别标签，形状为 (batch_size,)
    :return: 光谱相似性损失，标量
    """
    # 获取每个样本对应的类别原型
    batch_size = unet_output.shape[0]
    prototype_batch = prototypes[labels]  # 形状为 (batch_size, C, H, W)

    # 计算光谱相似性损失（例如，均方误差）
    similarity_loss = F.mse_loss(unet_output, prototype_batch, reduction='mean')
    
    return similarity_loss

def train_loop(class_weights,config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    # 初始化加速器和张量板日志
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs").replace('\\','/'),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        # accelerator.init_trackers("diffusion_train")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # 加载并转换 prototypes
    prototypes = np.load('DataArray/Map_Prototypes.npy')  # (16, 4, 32, 32)
    prototypes = torch.tensor(prototypes).float().to(accelerator.device)  # 转换为 PyTorch 张量并移动到合适的设备

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images,labels = batch
            # labels=torch.nn.functional.one_hot(labels,num_classes=16)
            # Sample noise to add to the images
            clean_images, labels = clean_images.to(accelerator.device), labels.to(accelerator.device)

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False,class_labels=labels)[0]

                losses = F.mse_loss(noise_pred, noise, reduction='none').mean((1, 2, 3))  # 对非批量维度求均值
                weighted_losses = losses * class_weights[labels.long()]  # 应用每个样本的权重
                loss = weighted_losses.mean()  # 求所有样本的平均
                        # 计算光谱相似性损失
                spectral_loss = compute_spectral_similarity_loss(noise_pred, prototypes, labels.long())
                # 合并损失
                total_loss = loss + config.spectral_loss_weight * spectral_loss
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if accelerator.is_main_process:
            pipeline = LabeledDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
    return model, noise_scheduler


def unsupervised_train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    # 初始化加速器和张量板日志
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs").replace('\\','/'),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        # accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Unsupervised Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images,labels = batch
            #构造和labels相同形状的假标签，值全为-1
            fake_labels = -1 * torch.ones_like(labels)
            # labels=torch.nn.functional.one_hot(labels,num_classes=16)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False,class_labels=fake_labels)[0]#无监督训练，不传入标签
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if accelerator.is_main_process:
            pipeline = LabeledDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
    return model, noise_scheduler