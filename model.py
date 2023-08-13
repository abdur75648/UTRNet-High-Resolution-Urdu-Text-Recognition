"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from modules.feature_extraction import HRNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.dropout_layer import dropout_layer
from modules.prediction import Attention
import torch.nn as nn

# Other CNN Architectures
from modules.feature_extraction import DenseNet_FeatureExtractor, InceptionUNet_FeatureExtractor
from modules.feature_extraction import RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.feature_extraction import ResUnet_FeatureExtractor, AttnUNet_FeatureExtractor
from modules.feature_extraction import UNet_FeatureExtractor, UNetPlusPlus_FeatureExtractor
from modules.feature_extraction import VGG_FeatureExtractor

# Other sequential models
from modules.sequence_modeling import LSTM, GRU, MDLSTM

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling,
                       'Pred': opt.Prediction}

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'HRNet':
            self.FeatureExtraction = HRNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'Densenet':
            self.FeatureExtraction = DenseNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'InceptionUnet':
            self.FeatureExtraction = InceptionUNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResUnet':
            self.FeatureExtraction = ResUnet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'AttnUNet':
            self.FeatureExtraction = AttnUNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'UNet':
            self.FeatureExtraction = UNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'UnetPlusPlus':
            self.FeatureExtraction = UNetPlusPlus_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1)) # Transform final (imgH/16-1) -> 1
        
        """
        Temporal Dropout
        """
        self.dropout1 = dropout_layer(opt.device)
        self.dropout2 = dropout_layer(opt.device)
        self.dropout3 = dropout_layer(opt.device)
        self.dropout4 = dropout_layer(opt.device)
        self.dropout5 = dropout_layer(opt.device)

        """ Sequence modeling"""
        if opt.SequenceModeling == 'LSTM':
            self.SequenceModeling = LSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size)
        elif opt.SequenceModeling == 'GRU':
            self.SequenceModeling = GRU(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size)
        elif opt.SequenceModeling == 'MDLSTM':
            self.SequenceModeling = MDLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size)
        elif opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size)
        elif opt.SequenceModeling == 'DBiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        else:
            raise Exception('No Sequence Modeling module specified')
        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class, opt.device)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text=None, is_train=True):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # print(visual_feature.shape) # [32, 32, 32, 400] #HRNet, [32, 512, 32, 400] #UNet
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # print(visual_feature.shape) # [32, 400, 32, 1] #HRNet, [32, 400, 512, 1] #UNet
        visual_feature = visual_feature.squeeze(3)
        # print(visual_feature.shape) # [32, 400, 32] #HRNet, [32, 400, 512] #UNet

        """ Temporal Dropout + Sequence modeling stage """
        # contextual_feature = self.SequenceModeling(visual_feature) ##### Without temporal dropout
        if (self.training):
            visual_feature_after_dropout1 = self.dropout1(visual_feature)
            contextual_feature = self.SequenceModeling(visual_feature_after_dropout1)
        else :
            visual_feature_after_dropout1 = self.dropout1(visual_feature)
            visual_feature_after_dropout2 = self.dropout2(visual_feature)
            visual_feature_after_dropout3 = self.dropout3(visual_feature)
            visual_feature_after_dropout4 = self.dropout4(visual_feature)
            visual_feature_after_dropout5 = self.dropout5(visual_feature)
            contextual_feature1 = self.SequenceModeling(visual_feature_after_dropout1)
            contextual_feature2 = self.SequenceModeling(visual_feature_after_dropout2)
            contextual_feature3 = self.SequenceModeling(visual_feature_after_dropout3)
            contextual_feature4 = self.SequenceModeling(visual_feature_after_dropout4)
            contextual_feature5 = self.SequenceModeling(visual_feature_after_dropout5)
            contextual_feature =  ( (contextual_feature1).add ((contextual_feature2).add(((contextual_feature3).add(((contextual_feature4).add(contextual_feature5)))))) ) * (1/5)

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            if text is None:
                raise Exception('Input text (for prediction) to model is None')
            text = text.to(self.opt.device)
            prediction = self.Prediction(contextual_feature, text, is_train, batch_max_length=self.opt.batch_max_length)
        
        return prediction
