"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import time
import random
import argparse

import torch
import time
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.nn.init as init
import torch.optim as optim

from model import Model
from test import validation
from utils import Averager, Logger
from dataset import hierarchical_dataset, AlignCollate
from utils import CTCLabelConverter, AttnLabelConverter

def train(opt, device):
    # opt.num_gpu = 8 # Multi-GPU Training -> Uncomment line 24 & line 63-64
    logger = Logger(f'./saved_models/{opt.exp_name}/log_train.txt')
    opt.device = device
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    # Load The Train Dataset
    AlignCollate_train = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    train_dataset, train_dataset_log  = hierarchical_dataset(root=opt.train_data, opt=opt,rand_aug=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_train, pin_memory=False)
    # """
    logger.log('Device : {}'.format(device))
    logger.log('-' * 80 + '\n')
    logger.log("Training Dataset Loaded: ", train_dataset_log)

    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt, rand_aug=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=False)
    logger.log("Validation Dataset Loaded : ", valid_dataset_log)
    logger.log('-' * 80 + '\n')
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    # Multi-GPU Training -> Uncomment line 24 & line 63-64
    # if opt.num_gpu > 1:
    #     model = torch.nn.DataParallel(model, device_ids=range(opt.num_gpu))
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            logger.log(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:
            if 'weight' in name:
                param.data.fill_(1)
            continue

    model = model.to(device)
    model.train()
    if opt.saved_model != '':
        logger.log(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    logger.log("Model:")
    logger.log(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    sum_params_num = sum(params_num)
    logger.log(f'Trainable params num : {sum_params_num/1000000:.2f}M')
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    logger.log("Optimizer: ", optimizer)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    reduce_lr = [50,100] # epochs where you want to reduce the LR by 10
    
    # Slowly decrease the learning rate from initial lr to 0 over the course of training
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=0.01)

    """ final options """
    # print(opt)
    opt_log = '------------ Options -------------\n'
    args = vars(opt)
    for k, v in args.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    logger.log(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            logger.log(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    
    init_time = time.time()
    for epoch in tqdm(range(opt.num_epochs)):
        logger.log("="*20,"Epoch =",epoch+1,"="*20)
        
        for i, (image_tensors, labels) in enumerate(tqdm(train_loader)):
            #image_tensors, labels = train_dataset_batch.get_batch()
            image = image_tensors.to(device)
            # Cheking if correct inputs are being given)
            # if os.path.exists("check_dir"):
            #     import shutil
            #     shutil.rmtree("check_dir")
            # os.makedirs("check_dir")
            # for var1 in range(image.size(0)):
            #     file_name_1 = "check_dir/" + str(iteration)+str(var1)+"label.txt"
            #     with open(file_name_1,"a", encoding="utf-8") as labelFile:
            #         labelFile.write((labels[var1]))
            #     img_normalized = image[var1]*0.5 + 0.5
            #     pilTrans = transforms.ToPILImage()(img_normalized).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            #     pilTrans.save(str("check_dir/" + str(iteration)+str(var1)+"image.png"))

            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text=text[:, :-1].to(device))  # align with Attention.forward
                target = text[:, 1:].to(device)  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]).contiguous(), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)
        
        if epoch in reduce_lr:
            scheduler.step()
            logger.log(f'Learning rate reduced to {scheduler.get_lr()}')
        
        # scheduler.step()

        elapsed_time = time.time() - start_time
        model.eval()
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, _ = validation(
                model, criterion, valid_loader, converter, opt, device)
        model.train()
        # training loss and validation loss
        loss_log = f'[{epoch+1}/{opt.num_epochs}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        loss_avg.reset()
        current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.2f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

        # keep best accuracy model (on valid dataset)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            # torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
        if current_norm_ED > best_norm_ED:
            best_norm_ED = current_norm_ED
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
        best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.4f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'
        loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
        logger.log(loss_model_log + '\n')
    
    end_time = time.time()
    logger.log("Total time taken for training: " + str(end_time-init_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True ,help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
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
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ GPU Selection """
    parser.add_argument('--device_id', type=str, default=None, help='cuda device ID')
    
    opt = parser.parse_args()
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    file = open("UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "
    file.close()
    
    """ Seed setting """
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    
    cuda_str = 'cuda'
    if opt.device_id is not None:
        cuda_str = f'cuda:{opt.device_id}'
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    train(opt, device)