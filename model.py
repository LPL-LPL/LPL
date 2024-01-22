import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class LPL(nn.Module):

    def __init__(self, args, base_encoder):  # base_encoder SupConResNet
        super().__init__()
        
        pretrained = args.dataset == 'cub200'
        # we allow pretraining for CUB200, or the network will not converge
        
        if args.synthetic_data == 'cifar80no': 
            num_class=80  
        elif args.synthetic_data == 'cifar100nc':
            num_class=100
        else:  
             num_class=args.n_classes

        self.encoder_q = base_encoder(num_class=num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue  
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))  
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(num_class,args.low_dim))
        self.queue = F.normalize(self.queue, dim=0)  

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)   

        # replace the keys at ptr (dequeue and enqueue)  
        if ptr + batch_size >=args.moco_queue:
            self.queue[ptr:, :] = keys[:args.moco_queue-ptr]
            self.queue_pseudo[ptr:] = labels[ : args.moco_queue-ptr]
            self.queue[: batch_size +ptr - args.moco_queue , :] = keys[args.moco_queue-ptr:]
            self.queue_pseudo[: batch_size +ptr - args.moco_queue ] = labels[args.moco_queue-ptr:]
        else:
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

        
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_all = x.shape[0]
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x, idx_unshuffle


    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, update=False):
        
        
        # test
        if eval_only == True:
            output, q = self.encoder_q(img_q)
            logits = output
            return {'logits': logits}        
        
        # warmup forward-------------------------------------------------------------------------------------------
        if partial_Y==None:
            output, q = self.encoder_q(img_q)
            logits = output
            output2, q = self.encoder_q(im_k)
            logits2 = output2
            return {'logits': logits, 'logits2': logits2}
        
            
        
        # LPL forward-------------------------------------------------------------------------------------------
        else:
            output, q = self.encoder_q(img_q)
            

            if eval_only:
                return output
            
            # # for testing
            # predicted_scores = torch.softmax(output.clone().detach(), dim=1) * partial_Y
            # max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)  

            temperature=0.01
            predicted_scores = torch.softmax((output.clone().detach() * partial_Y)/temperature, dim=1)
            max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)

            # compute protoypical logits
            prototypes = self.prototypes
            logits_prot = torch.mm(q.clone().detach(), prototypes.t())  
            score_prot = torch.softmax(logits_prot, dim=1)  
            # print(self.prototypes.size(),q.size()) 
            
    

            # update momentum prototypes with pseudo labels
            for feat, label in zip(q.clone().detach(), pseudo_labels_b):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
            
            # normalize prototypes    
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1) 
        
            # compute key features 
            with torch.no_grad():  # no gradient 
                self._momentum_update_key_encoder(args)  # update the momentum encoder

                output2, k = self.encoder_k(im_k) 
        
            features = torch.cat((q, k, self.queue), dim=0) 
            pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)  
            # to calculate SupCon Loss using pseudo_labels

        
            # dequeue and enqueue
            if update:
                self._dequeue_and_enqueue(k, pseudo_labels_b, args)

            return output, output2 ,features, pseudo_labels, score_prot


