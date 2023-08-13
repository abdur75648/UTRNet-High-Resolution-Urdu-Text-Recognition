"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import pytz
import math
import argparse
from PIL import Image
from datetime import datetime

import torch
import torch.utils.data

from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter, Logger

def read(opt, device):
    opt.device = device
    os.makedirs("read_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'read_outputs/{datetime_now}.txt')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    logger.log('Loaded pretrained model from %s' % opt.saved_model)
    model.eval()
    
    if opt.rgb:
        img = Image.open(opt.image_path).convert('RGB')
    else:
        img = Image.open(opt.image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape) # torch.Size([1, 1, 32, 400])
    batch_size = img.shape[0] # 1
    img = img.to(device)
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    logger.log(preds_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='path to image to read')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=100, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, default="HRNet", #required=True,
                        help='FeatureExtraction stage VGG|RCNN|ResNet|UNet|HRNet|Densenet|InceptionUnet|ResUnet|AttnUNet|UNet|VGG')
    parser.add_argument('--SequenceModeling', type=str, default="DBiLSTM", #required=True,
                        help='SequenceModeling stage LSTM|GRU|MDLSTM|BiLSTM|DBiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", #required=True,
                        help='Prediction stage CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ GPU Selection """
    parser.add_argument('--device_id', type=str, default=None, help='cuda device ID')
    
    opt = parser.parse_args()
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    """ vocab / character number configuration """
    file = open("UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "
    
    cuda_str = 'cuda'
    if opt.device_id is not None:
        cuda_str = f'cuda:{opt.device_id}'
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    read(opt, device)