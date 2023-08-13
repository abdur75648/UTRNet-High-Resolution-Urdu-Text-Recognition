"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch.nn as nn
import torch
import numpy as np

class dropout_layer(nn.Module):
    def __init__(self,device):
        super(dropout_layer, self).__init__()
        self.device = device
    def forward(self, input):
        nums = (np.random.rand(input.shape[1]) > 0.2).astype (int)
        dummy_array_output = torch.from_numpy(nums).to(self.device)
        dummy_array_output_t = torch.reshape(dummy_array_output, (input.shape[1], 1)).to(self.device) #Transpose
        dummy_array_output_f = dummy_array_output_t.repeat(input.shape[0], 1,input.shape[2]).to(self.device) #Same size as input
        output =  input*dummy_array_output_f  #element-wise multiplication
        return output