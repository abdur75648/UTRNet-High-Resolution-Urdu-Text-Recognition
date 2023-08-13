"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import math
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from model import Model
import matplotlib.pyplot as plt
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter

def main(opt, device):
    opt.rgb = False
    opt.device = device
    # opt.vis_dir does not exist, make it
    if os.path.exists(opt.vis_dir):
        import shutil
        shutil.rmtree(opt.vis_dir)
    os.mkdir(opt.vis_dir)
    # Loading image
    if opt.image_path is None:
        raise Exception("Please provide image path for feature visualization")
    if opt.rgb:
        img = Image.open(opt.image_path).convert('RGB')
    else:
        img = Image.open(opt.image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(32 * ratio) > 400:
        resized_w = 400
    else:
        resized_w = math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, 32, 400))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape) # torch.Size([1, 1, 32, 400])
    img = img.to(device)
    img.requires_grad = True
    
    # Making model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = model.to(device)
    
    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print('Loaded pretrained model from %s' % opt.saved_model)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.train()
    
    preds = model(img)

    preds_size = torch.IntTensor([preds.size(1)] * 1)
    score, preds_index = preds.max(2)
    # preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    # print(preds_str)
    preds_index = preds_index.squeeze(0)
    arr_is_consecutive_duplicate = []
    # Store index of all consecutive duplicates present in preds_index
    for i in range(len(preds_index) - 1):
        if preds_index[i] == preds_index[i + 1]:
            arr_is_consecutive_duplicate.append(i)
    len(arr_is_consecutive_duplicate)
    # Remove all consecutive duplicates and blank characters
    indices_final = []
    for i in range(preds_index.shape[0]):
        if preds_index[i] != 0 and i not in arr_is_consecutive_duplicate:
            indices_final.append(i)
            
    # For each predicted character individually
    plt.figure(figsize=(100, 100))
    for i,index in enumerate(indices_final):
        score[0][index].backward(retain_graph=True)
        slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
        # Reset img.grad
        img.grad = None
        slc = (slc - slc.min())/(slc.max()-slc.min())
        input_img = img[0]*0.5 + 0.5
        plt.subplot(len(indices_final), 2, 2*i+1)
        plt.imshow(np.transpose(input_img.detach().cpu().numpy(), (1, 2, 0))[:,::-1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(len(indices_final), 2, 2*i+2)
        plt.imshow(slc.detach().cpu().numpy()[:,::-1], cmap=plt.cm.hot)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{opt.vis_dir}/char_wise.png")
    
    # For all - final
    plt.figure(figsize=(100, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(np.transpose(input_img.detach().cpu().numpy(), (1, 2, 0))[:,::-1]) #, cmap='gray')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i,index in enumerate(indices_final):
        score[0][index].backward(retain_graph=True)
    slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
    slc = (slc - slc.min())/(slc.max()-slc.min())
    input_img = img[0]*0.5 + 0.5
    plt.subplot(2, 1, 2)
    plt.imshow(slc.detach().cpu().numpy()[:,::-1], cmap=plt.cm.hot)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{opt.vis_dir}/overall.png")

   
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_dir', type=str, default='vis_salency_maps', help='path to save visualization')
    parser.add_argument('--image_path', type=str, default=None, help='path to image for feature map visualization')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage')
    parser.add_argument('--saved_model', type=str, required=True, help='path to saved_model to evaluation')
    parser.add_argument('--input_channel', type=int, default=1, help='Number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='Number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of the BiLSTM hidden state')
    opt = parser.parse_args()
    """ vocab / character number configuration """
    file = open("UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    main(opt, device)