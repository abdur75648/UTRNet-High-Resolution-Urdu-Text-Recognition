"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import pytz
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable

import os,random,shutil
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            
            try:
                text = [self.dict[char] for char in text]
            except KeyError as e:
                continue
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


def imshow(img, title,batch_size=1):
  std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
  mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
  npimg = np.multiply(img.numpy(), std_correction) + mean_correction
  plt.figure(figsize = (batch_size * 4, 4))
  plt.axis("off")
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.title(title)
  plt.show()


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class Logger(object):
    """For logging while training"""
    def __init__(self, path):
        self.logFile = path
        datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
        with open(self.logFile,"w",encoding="utf-8") as f:
            f.write("Logging at @ " + str(datetime_now) + "\n")

    def log(self,*input):
        message = ""
        for x in input:
            message+=str(x) + " "
        message = message.strip()
        print(message)
        with open(self.logFile,"a",encoding="utf-8") as f:
            f.write(str(message)+"\n")


def allign_two_strings(x:str, y:str, pxy:int=1, pgap:int=1):
    """
    Source: https://www.geeksforgeeks.org/sequence-alignment-problem/
    """
    i = 0
    j = 0
    m = len(x)
    n = len(y)
    dp = np.zeros([m+1,n+1], dtype=int)
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
 
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + pxy,
                                dp[i - 1][j] + pgap,
                                dp[i][j - 1] + pgap)
            j += 1
        i += 1
     
    l = n + m 
    i = m
    j = n
     
    xpos = l
    ypos = l
 
    xans = np.zeros(l+1, dtype=int)
    yans = np.zeros(l+1, dtype=int)
 
    while not (i == 0 or j == 0):
        #print(f"i: {i}, j: {j}")
        if x[i - 1] == y[j - 1]:       
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
        elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:
         
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
         
        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
        elif (dp[i][j - 1] + pgap) == dp[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1
         
 
    while xpos > 0:
        if i > 0:
            i -= 1
            xans[xpos] = ord(x[i])
            xpos -= 1
        else:
            xans[xpos] = ord('_')
            xpos -= 1
     
    while ypos > 0:
        if j > 0:
            j -= 1
            yans[ypos] = ord(y[j])
            ypos -= 1
        else:
            yans[ypos] = ord('_')
            ypos -= 1

    id = 1
    i = l
    while i >= 1:
        if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
            id = i + 1
            break
         
        i -= 1
 
    i = id
    x_seq = ""
    while i <= l:
        x_seq += chr(xans[i])
        i += 1
 
    # Y
    i = id
    y_seq = ""
    while i <= l:
        y_seq += chr(yans[i])
        i += 1
    
    return x_seq, y_seq

# Function to count the number of trainable parameters in a model in "Millions"
def count_parameters(model,precision=2):
    return (round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 10.**6, precision))

'''
# Code for counting the number of FLOPs in the CNN backbone during inference
Source - https://github.com/fdbtrs/ElasticFace/blob/main/utils/countFLOPS.py
'''

def count_model_flops(model,in_channels=1, input_res=[32, 400], multiply_adds=True):
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)
    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is not None:
            bias_ops = self.bias.nelement() if self.bias.nelement() else 0
            flops = batch_size * (weight_ops + bias_ops)
        else:
            flops = batch_size * weight_ops
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        # If kernel_size is a tuple type, computer ops as product of elements or else if it is int type, compute ops as square of kernel_size
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] if isinstance(self.kernel_size, tuple) else self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)
    
    def dropout_hook(self, input, output):
        # calculate the number of operations for a dropout function by assuming that each operation involves one comparison and one multiplication
        batch_size, input_channels, input_height, input_width = input[0].size()
        list_conv.append(2*batch_size*input_channels*input_height*input_width)
    
    def sigmoid_hook(self,input,output):
        # calculate the number of operations for a sigmoid function by assuming that each operation involves two multiplications and one addition
        batch_size, input_channels, input_height, input_width = input[0].size()
        list_conv.append(3*batch_size*input_channels*input_height*input_width)
    
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.scale_factor * self.scale_factor # * (self.in_channels / self.groups)
        flops = (kernel_ops * (
            2 if multiply_adds else 1)) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)

    handles = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                handles.append(net.register_forward_hook(conv_hook))
            elif isinstance(net, torch.nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            elif isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm1d):
                handles.append(net.register_forward_hook(bn_hook))
            elif isinstance(net, torch.nn.ReLU) or isinstance(net, torch.nn.PReLU):
                handles.append(net.register_forward_hook(relu_hook))
            elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            elif isinstance(net, torch.nn.Dropout):
                handles.append(net.register_forward_hook(dropout_hook))
            elif isinstance(net,torch.nn.Upsample):
                handles.append(net.register_forward_hook(upsample_hook))
            elif isinstance(net,torch.nn.Sigmoid):
                handles.append(net.register_forward_hook(sigmoid_hook))
            else:
                print("warning" + str(net))
            return
        for c in childrens:
            foo(c)

    model.eval()
    foo(model)
    input = Variable(torch.rand(in_channels, input_res[1], input_res[0]).unsqueeze(0), requires_grad=True)
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    for h in handles:
        h.remove()
    model.train()
    
    def flops_to_string(flops, units='MFLOPS', precision=4):
        if units == 'GFLOPS':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MFLOPS':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KFLOPS':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPS'
    
    return flops_to_string(total_flops)


def draw_feature_map(visual_feature,vis_dir,num_channel=10):
    """draws feature maps for the given visual features
    Args:
        visual_feature (Tensor): Shape (C, H, W)
        vis_dir (String): Directory to save the feature maps
    """
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    # Save visual_feature from num_channel random channels for visualization
    for i in range(num_channel):
        random_channel = random.randint(0, visual_feature.shape[1]-1)
        visual_feature_for_visualization = visual_feature[0, random_channel, :, :].detach().cpu().numpy()
        # Horizontal flip
        visual_feature_for_visualization = visual_feature_for_visualization[:,::-1]
        # Normalize
        visual_feature_for_visualization = (visual_feature_for_visualization - visual_feature_for_visualization.min()) / (visual_feature_for_visualization.max() - visual_feature_for_visualization.min())
        # Draw heatmap
        plt.imshow(visual_feature_for_visualization, cmap='gray', interpolation='nearest')
        plt.axis("off")
        plt.savefig(os.path.join(vis_dir, "channel_{}.png".format(random_channel)), bbox_inches='tight', pad_inches=0)