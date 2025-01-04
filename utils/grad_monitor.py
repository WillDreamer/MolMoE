import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class GradientMonitor:
    def __init__(self, model, writer: SummaryWriter, log_freq=100):
        self.model = model
        self.writer = writer
        self.log_freq = log_freq
        self.step = 0
        
    def log_gradients(self):
        """记录梯度信息到tensorboard"""
        if self.step % self.log_freq != 0:
            self.step += 1
            return
            
        # 记录每层的梯度范数
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 计算梯度范数
                grad_norm = param.grad.norm().item()
                
                # 记录到tensorboard
                self.writer.add_scalar(f'gradients/{name}_norm', 
                                     grad_norm,
                                     self.step)
                
                # 记录梯度分布
                self.writer.add_histogram(f'gradients/{name}_dist',
                                        param.grad.detach().cpu().numpy(),
                                        self.step)
                
                # 打印大梯度警告
                if grad_norm > 10.0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f} in layer {name}")
                    
        # 计算整体梯度范数
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.writer.add_scalar('gradients/total_norm', total_norm, self.step)
        
        if total_norm > 10.0:
            print(f"Warning: Total gradient norm is {total_norm:.2f}")
            
        self.step += 1

    def log_parameter_norms(self):
        """记录模型参数范数"""
        if self.step % self.log_freq != 0:
            return
            
        for name, param in self.model.named_parameters():
            self.writer.add_scalar(f'parameters/{name}_norm',
                                 param.norm().item(),
                                 self.step) 