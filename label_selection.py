import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import pickle
import os


class label_selection_mdel:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):
        self.num_classes = num_classes
        device = torch.device('cuda:%s' % config.gpu) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.tmp_epoch = 0
        self.dataset = config.dataset
        self.momentum = config.labelsmomentum
        

        # Thresholds for different subsets
        self.adjust_lr = 1
        self.lamd_ce = 1
        self.lamd_h = 1
        self.sigma = 0.5
       
        if 'cifar' in config.dataset:
            N = 50000
        elif 'clothing1M' in config.dataset:
            N = 1037498
        elif 'tiny_imagenet' in config.dataset:
            N = 100000
        elif 'food' in config.dataset:
            N = 310009  # 
        elif 'Animal' in config.dataset:
            N = 50000
        elif 'webvision' in config.dataset:
            N = 65944
        elif 'web-bird' in config.dataset:
            N = 18388
        elif 'web-car' in config.dataset:
            N = 21448
        elif 'web-aircraft' in config.dataset:
            N = 13503
        self.N = N
        
        # Variable definition
        self.s_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.w_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.ws_prev_confidence = torch.ones(N).to(self.device)*1/N

        self.w_probs = torch.zeros(N, num_classes).to(self.device)
        self.s_probs = torch.zeros(N, num_classes).to(self.device)
        self.labels = torch.ones(N).long().to(self.device)
        self.weak_labels = self.labels.detach().clone()

        self.clean_flags = torch.zeros(N).bool().to(self.device)
        self.hard_flags = torch.zeros(N).bool().to(self.device)
        self.correction_flags = torch.zeros(N).bool().to(self.device)
        self.weak_flags = torch.zeros(N).bool().to(self.device)
        self.w_selected_flags = torch.zeros(N).bool().to(self.device)
        self.s_selected_flags = torch.zeros(N).bool().to(self.device)
        self.selected_flags = torch.zeros(N).bool().to(self.device)
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.acc_list = list()
        self.num_list = list()
        

    def train(self, indexes,w_logits,s_logits):
        b_clean_flags = self.clean_flags[indexes]
        clean_num = b_clean_flags.sum()
        b_hard_flags = self.hard_flags[indexes]
        hard_num = b_hard_flags.sum()                
                    
        with torch.no_grad():
            w_prob = F.softmax(w_logits, dim=1)
            self.w_probs[indexes] = w_prob
            s_prob = F.softmax(s_logits, dim=1)
            self.s_probs[indexes] = s_prob


    def get_clean_labels(self,train_noisy_labels,device,forget_rate=0.2):
        with torch.no_grad():

            ws_probs = (self.w_probs+self.s_probs)/2
            ws_prob_max, ws_label= torch.max(ws_probs, dim=1) 
            small_loss= F.cross_entropy(ws_probs.to(device), train_noisy_labels.to(device), reduce = False)
            sorted_indices = torch.argsort(small_loss)
            percentage = 1.0-forget_rate
            class_indices_list = []
            class_counts = {}
            noisy_class_indices_list = []
            unique_labels = train_noisy_labels.unique()

            for label in unique_labels:
                class_indices = (train_noisy_labels == label).nonzero().squeeze()
                class_losses = small_loss[class_indices]
                _, indices = torch.topk(class_losses, int(percentage * len(class_losses)), largest=False)
                _, noisy_indices = torch.topk(class_losses, int(forget_rate * len(class_losses)), largest=True)
                class_indices_list.extend(class_indices[indices.cpu()].tolist())
                noisy_class_indices_list.extend(class_indices[noisy_indices.cpu()].tolist())
            class_indices_tensor = torch.tensor(class_indices_list)      
            noisy_class_indices_tensor = torch.tensor(noisy_class_indices_list)  

        self.clean_labels_index = class_indices_tensor.long()
        self.noisy_labels_index = noisy_class_indices_tensor.long()

