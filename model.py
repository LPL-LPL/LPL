import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class PLLNL(nn.Module):

    def __init__(self, args, base_encoder):  
        super().__init__()
        
        pretrained = args.dataset == 'cub200'

        
        if args.synthetic_data == 'cifar80no': 
            num_class=80 
        elif args.synthetic_data == 'cifar100nc':
            num_class=100
        else: 
             num_class=args.n_classes

        self.encoder_q = base_encoder(num_class=num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        self.encoder_k = base_encoder(num_class=num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False 


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

        

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, update=False):

        if eval_only==True:
            output, q = self.encoder_q(img_q)
            logits = output      
            return {'logits': logits}              

        elif partial_Y==None:
            output, q = self.encoder_q(img_q)
            logits = output
            output2, q = self.encoder_q(im_k)
            logits2 = output2
            return {'logits': logits, 'logits2': logits2}                            

        else:
            output, q = self.encoder_q(img_q)       
            if eval_only:
                return output
            temperature=0.1
            predicted_scores = torch.softmax((output.clone().detach() * partial_Y)/temperature, dim=1)
            max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)  
            prototypes = self.prototypes
            logits_prot = torch.mm(q.clone().detach(), prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)            

            for feat, label in zip(q.clone().detach(), pseudo_labels_b):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat  
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)        

            with torch.no_grad(): 
                self._momentum_update_key_encoder(args) 
                output2, k = self.encoder_k(im_k)       

            features = torch.cat((q, k, self.queue), dim=0) 
            pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0) 

            if update:
                self._dequeue_and_enqueue(k, pseudo_labels_b, args)

            return output, output2 ,features, pseudo_labels, score_prot


