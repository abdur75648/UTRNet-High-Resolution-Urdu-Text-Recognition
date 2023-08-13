"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os,shutil
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, Logger
from dataset import hierarchical_dataset, AlignCollate
from model import Model

def validation(model, criterion, evaluation_loader, converter, opt, device):
    """ validation or evaluation """
    eval_arr = []
    sum_len_gt = 0

    n_correct = 0
    
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(tqdm(evaluation_loader)):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image)
            forward_time = time.time() - start_time
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            preds = model(image, text=text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :].to(device)
            target = text_for_loss[:, 1:].to(device)  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                ED = 0
            elif len(gt) > len(pred):
                ED = 1 - edit_distance(pred, gt) / len(gt)
            else:
                ED = 1 - edit_distance(pred, gt) / len(pred)

            eval_arr.append([gt,pred,ED])

            sum_len_gt += len(gt)
            norm_ED += (ED*len(gt))

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(sum_len_gt)

    return valid_loss_avg.val(), accuracy, norm_ED, eval_arr


def test(opt, device):
    opt.device = device
    os.makedirs("test_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'test_outputs/{datetime_now}.txt')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    logger.log('Loaded pretrained model from %s' % opt.saved_model)
    # logger.log(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)#, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt, rand_aug=False)
        logger.log(eval_data_log)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy, norm_ED, eval_arr = validation( model, criterion, evaluation_loader, converter, opt,device)
    logger.log("="*20)
    logger.log(f'Accuracy : {accuracy:0.4f}\n')
    logger.log(f'Norm_ED : {norm_ED:0.4f}\n')
    logger.log("="*20)
    
    if opt.visualize:
        logger.log("Threshold - ", opt.threshold)
        logger.log("ED","\t","gt","\t","pred")
        arr = []
        for gt,pred,ED in eval_arr:
            ED = ED*100.0
            arr.append(ED)
            if ED<=(opt.threshold):
                logger.log(ED,"\t",gt,"\t",pred)
        plt.hist(arr, edgecolor="red")
        plt.savefig('test_outputs/'+str(datetime_now)+".png")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='for visualization of bad samples')
    parser.add_argument('--threshold', type=float, help='Save samples below this threshold in txt file', default=50.0)
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
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
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ GPU Selection """
    parser.add_argument('--device_id', type=str, default=None, help='cuda device ID')

    opt = parser.parse_args()
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    
    # Fix random seeds for both numpy and pytorch
    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    # opt.eval_data = "/DATA/parseq/val/"
    # test(opt, device)
    
    # opt.eval_data = "/DATA/parseq/IIITH/lmdb_new/"
    # test(opt, device)
    
    # opt.eval_data = "/DATA/public_datasets/UPTI/valid/"
    # test(opt, device)
    
    test(opt, device)