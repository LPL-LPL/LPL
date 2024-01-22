import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy


class partial_loss(nn.Module):
    """PICO: CONTRASTIVE LABEL DISAMBIGUATION FOR PARTIAL LABEL LEARNING: 
        https://doi.org/10.48550/arXiv.2201.08984."""
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):  
        logsm_outputs = F.log_softmax(outputs, dim=1)


        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            # # onehot
            # _, prot_pred = (temp_un_conf * batchY).max(dim=1)  
            # pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()  
            
            # # # softmax
            pseudo_label=torch.softmax(temp_un_conf* batchY/0.01 ,-1) # temperature==0.001    

            # self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
            #      + (1 - self.conf_ema_m) * pseudo_label   # torch.Size([128, 100]) 

        return None

class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None: 

            mask = mask.float().detach().to(device)  
      
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)

            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  
            logits = anchor_dot_contrast - logits_max.detach()  



            logits_mask = torch.scatter(
                torch.ones_like(mask),  
                1,
                torch.arange(batch_size).view(-1, 1).to(device),  
                0
            )  

            mask = mask * logits_mask 



            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)       
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
    

            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) 
            l_neg = torch.einsum('nc,kc->nk', [q, queue])  
            logits = torch.cat([l_pos, l_neg], dim=1)  

            logits /= self.temperature  

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda() 
            loss = F.cross_entropy(logits, labels)

        return loss