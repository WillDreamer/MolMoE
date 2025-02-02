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
        if self.step % self.log_freq != 0:
            self.step += 1
            return
            
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                self.writer.add_scalar(f'gradients/{name}_norm', 
                                     grad_norm,
                                     self.step)
                
                self.writer.add_histogram(f'gradients/{name}_dist',
                                        param.grad.detach().cpu().numpy(),
                                        self.step)
                
                if grad_norm > 10.0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f} in layer {name}")

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
        if self.step % self.log_freq != 0:
            return
            
        for name, param in self.model.named_parameters():
            self.writer.add_scalar(f'parameters/{name}_norm',
                                 param.norm().item(),
                                 self.step) 