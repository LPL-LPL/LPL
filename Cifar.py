import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
from model import PiCO
from net import *
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.builder import *
from utils.utils_algo import AverageMeter
from utils.loss import *
from data.image_folder import IndexedImageFolder
LOG_FREQ = 1
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100',"web-aircraft","web-bird",'web-car'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=["resnet18","resnet50",'sevenCNN'], 
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--warmuplr', '--warmup-learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial warmup learning rate', dest='warmuplr')
parser.add_argument('-lr_decay_epochs', type=str, default='100,150,180', 
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples') 
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight1', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--loss_weight2', default=0.5, type=float,
                    help='jsd loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')  
parser.add_argument('--prot_start', default=80, type=int, 
                    help = 'Start Prototype Updating')

parser.add_argument('--noise-type', type=str, default='symmetric') 
parser.add_argument('--closeset_ratio', type=float, default='0.2') 
parser.add_argument('--synthetic_data', type=str) 
parser.add_argument('--n_classes', type=int, default='100') 
parser.add_argument('--warmupepochs', type=int, default=1)  
parser.add_argument('--epsilon', type=float, default=0.5)  
parser.add_argument('--topk', type=int, default=10)  
parser.add_argument('--forget_rate', type=float, default=0.2) 
parser.add_argument('--rechange_per_epoch', type=int, default=50) 



def main():
    args = parser.parse_args() 
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')] 

    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')  

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    
    model_path = 'Cifar{time}ds_{ds}_lr_{lr}_ep_{ep}_ps_{ps}_lw_{lw}_warmupepochs_{warmupepochs}_rechange_per_epoch_{rechange_per_epoch}'.format(
                                            time=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), 
                                            ds=args.dataset,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            ps=args.prot_start,
                                            lw=args.loss_weight1,
                                            warmupepochs=args.warmupepochs,
                                            rechange_per_epoch=args.rechange_per_epoch)  
    args.exp_dir = os.path.join(args.exp_dir, model_path)  
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    

    model = PiCO(args, SupConResNet) 

    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    
    if args.synthetic_data == 'cifar80no': 
        num_classes = 80  
    elif args.synthetic_data == 'cifar100nc':
        num_classes = 100 
    else:
        num_classes = args.n_classes


    dataset, train_loader, test_loader = build_dataset_loader(args)
    topk_index=torch.zeros(dataset['n_train_samples'],args.topk).cuda()  
    train_givenY=torch.zeros(dataset['n_train_samples'],num_classes).cuda()  


    print('Calculating uniform targets...')
    

    train_noisy_labels = dataset['train'].noisy_labels
    train_noisy_labels = torch.tensor(train_noisy_labels) 
    train_givenY=F.one_hot(train_noisy_labels.cuda(),num_classes=num_classes) 

    

    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])  
    confidence = train_givenY.float()/tempY 
    confidence = confidence.cuda()


    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()

    if args.gpu==0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')


    with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
        f.write('args {}\n'.format(args))
        f.write('\n')

    best_acc = 0
    mmc = 0 
    for epoch in range(0, args.epochs):  
        is_best = False
        start_upd_prot = epoch>=args.prot_start

        adjust_learning_rate(args, optimizer, epoch)

        train_acc=train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, logger, topk_index,train_givenY,start_upd_prot)
        
        
        if epoch==args.warmupepochs-1:

            tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])  # torch.Size([50000, 100]) 0
            confidence = train_givenY.float()/tempY  # torch.Size([50000, 100])
            loss_fn = partial_loss(confidence)

        
        loss_fn.set_conf_ema_m(epoch, args)
        # reset phi

        acc_test = test(model, test_loader, args, epoch, logger)
        mmc = loss_fn.confidence.max(dim=1)[0].mean()
        


        if epoch==args.warmupepochs:
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write('    ----------  warmup end  -----------    \n')
        
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: TrainACC {}, Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                ,train_acc,acc_test, best_acc, optimizer.param_groups[0]['lr'], mmc))
        
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, tb_logger,topk_index,train_givenY,start_upd_prot=False):  
    

    if args.synthetic_data == 'cifar80no': 
        num_classes = 80 
    elif args.synthetic_data == 'cifar100nc':
        num_classes = 100 
    else:
        num_classes = args.n_classes
    
    # warmup  + PICO ---------------------------------------------------------------------------------------------------------------------------------------   

    train_loss = AverageMeter("train_loss")
    train_acc = AverageMeter("train_accuracy")
    topK_acc = AverageMeter("train_TopK")

    model.train()

    pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training')
    
    
    for it, sample in enumerate(pbar):

        
     
        #  warmup  ----------------------------------------------------------------------------------------------------------
        if epoch<args.warmupepochs:

            curr_lr = [group['lr'] for group in optimizer.param_groups][0]
            indices = sample['index']

            x, x_w, x_s = sample['data']
            x = x.to(args.gpu)
            x_s = x_s.to(args.gpu)

            y = sample['label'].to(args.gpu)

            output = model(x,x_s,update=True)
            
            logits = output['logits']
 
            probs = logits.softmax(dim=1) 
            index = torch.argmax(probs,dim=1)  

            given_labels = get_smoothed_label_distribution(y, num_classes, epsilon=args.epsilon)
            pnp_loss = cross_entropy(logits, given_labels, reduction='mean')

            
            train_acc1, accK = accuracy(logits, y, topk=(1, args.topk))
            train_acc.update(train_acc1[0])

        

        if(epoch==args.warmupepochs-1): 
            per_topk_prob, per_topk_index=torch.topk(probs,k=args.topk,dim=1,largest=True, sorted=True, out = None) 
            topk_index[indices]=per_topk_index.float()

            small_loss= F.cross_entropy(logits.to(args.gpu), y, reduce = False)
        
            ind_sorted = np.argsort(small_loss.data.cpu()).cuda()
            remember_rate = 1 - args.forget_rate
            num_remember = int(remember_rate * len(ind_sorted))
            ind_update=ind_sorted[:num_remember].detach()  
            for i in range(0,args.topk):
                topk_index[indices.cuda(args.gpu)[ind_update],i]=sample['label'].cuda(args.gpu)[ind_update].float().detach()
          

    
        # PICO training ---------------------------------------------------------------------------------------------------------------------------------------    

    
        end = time.time()
        index = sample["index"]  
        Y= train_givenY[index]  
        Y_true = sample['label']  


        indices = sample['index']
        if ((epoch-args.warmupepochs+1)%args.rechange_per_epoch==0 and epoch-args.warmupepochs+1>0):
            decrease=int((epoch-args.warmupepochs+1)//args.rechange_per_epoch)
            x, x_w, x_s = sample['data']
            x = x.to(args.gpu)
            x_s = x_s.to(args.gpu)

            y = sample['label'].to(args.gpu)

            with torch.no_grad():
                output = model(x,x_s) 

            logits = output['logits']
            logits2 = output['logits2']
            
            probs = logits.softmax(dim=1)
            index = torch.argmax(probs,dim=1)  
            per_topk_prob, per_topk_index=torch.topk(probs,k=args.topk,dim=1,largest=True, sorted=True, out = None) 
            topk_index[indices]=per_topk_index.float()
           

            small_loss=  F.cross_entropy(logits.to(args.gpu), y, reduce = False) 


            ind_sorted = np.argsort(small_loss.data.cpu()).cuda()
            remember_rate = 1 - args.forget_rate
            num_remember = int(remember_rate * len(ind_sorted)*(1-0.000*decrease)) 

            ind_update=ind_sorted[:num_remember]  
            for i in range(0,args.topk):
                topk_index[indices.cuda(args.gpu)[ind_update],i]=sample['label'].cuda(args.gpu)[ind_update].float()



        if epoch>=args.warmupepochs:
            x, x_w, x_s = sample['data']
            x, x_s = x.to(args.gpu), x_s.to(args.gpu)
                        
            cls_out, cls_out2,features_cont, pseudo_target_cont, score_prot = model(x, x_s, Y, args,update=True)

            batch_size = cls_out.shape[0]
            pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

            train_acc1= accuracy(cls_out, Y_true)

            train_acc.update(train_acc1[0])

            lam=np.random.beta(25,10)
            new_index = torch.randperm(x.size(0)).cuda()
            mix_x = lam*x + (1-lam)*x[new_index,:]
            mix_x_s = lam*x_s + (1-lam)*x_s[new_index,:]
            mix_Y = lam*Y + (1-lam)*Y[new_index]

            mix_cls_out,mix_cls_out2, mix_features_cont, mix_pseudo_target_cont, mix_score_prot = model(mix_x, mix_x_s, mix_Y, args )#, update=True
                
        
            if start_upd_prot:  
                loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)

            if start_upd_prot:  
                mix_pseudo_target_cont = mix_pseudo_target_cont.contiguous().view(-1, 1)
                mask1 = torch.eq(mix_pseudo_target_cont[:batch_size], mix_pseudo_target_cont.T).float().cuda()  
                margin=1
                mask2=torch.mm(mix_features_cont[:batch_size],mix_features_cont.T) 

                mask2=torch.where(mask2 > margin, 1, 0)


                mask=mask2+mask1
                mask=torch.where(mask >=1, 1, 0)

            else:
                mask = None

            loss_cont = loss_cont_fn(features=mix_features_cont, mask=mask, batch_size=batch_size)
        
            loss_cls = loss_fn(cls_out, index.type(torch.long))

            
            jocor_loss = loss_jocor(cls_out.to(args.gpu), cls_out2.to(args.gpu), Y_true.to(args.gpu))



        if epoch<args.warmupepochs:
            loss = pnp_loss 
        else:
            loss = loss_cls + loss_cont*args.loss_weight1 + jocor_loss*args.loss_weight2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if epoch==args.warmupepochs-1 :
        train_givenY.scatter_(1,topk_index.long(),1) 
        
    elif ((epoch-args.warmupepochs+1)%args.rechange_per_epoch==0 and epoch-args.warmupepochs+1>0) :
        train_givenY.scatter_(1,topk_index.long(),1) 
    



    return train_acc.avg


    

def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')       
        model.eval()    
        top1_acc = AverageMeter("Top1")
        topK_acc = AverageMeter("TopK")
        

        for _, sample in enumerate(tqdm(test_loader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            images = sample['data'].cuda()
            labels = sample['label'].cuda()  
            outputs = model(images,images,eval_only=True)["logits"] 

  
            acc1, accK = accuracy(outputs, labels, topk=(1, args.topk))
            top1_acc.update(acc1[0])
            topK_acc.update(accK[0])
        
        acc_tensors = torch.Tensor([top1_acc.avg,topK_acc.avg]).cuda(args.gpu)
        
        print('Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('TopK Acc', acc_tensors[1], epoch)             
    return acc_tensors[0]
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s 

def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))  
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)

def build_webfg_dataset(root, train_transform, test_transform):
    train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform) 
    test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}

def build_dataset_loader(params):
    if params.dataset.startswith('cifar'):
        transform = build_transform(rescale_size=32, crop_size=32)  
    
    if params.dataset.startswith('web-'):
        transform = build_transform(rescale_size=448, crop_size=448)

    if params.dataset == 'cifar100':
        if params.synthetic_data == 'cifar80no': 
            dataset = build_cifar100n_dataset(os.path.join("cifar100", params.dataset), CLDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), transform['cifar_test'], noise_type=params.noise_type, openset_ratio=0.2, closeset_ratio=params.closeset_ratio)  # 修改
        elif params.synthetic_data == 'cifar100nc':
            dataset = build_cifar100n_dataset(os.path.join("cifar100", params.dataset), CLDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), transform['cifar_test'], noise_type=params.noise_type, openset_ratio=0, closeset_ratio=params.closeset_ratio)  
    
    if params.dataset.startswith('web-'): 
        if params.dataset == 'web-bird':
            dataset = build_webfg_dataset(os.path.join("/data/fgcv/web-data/", "web-bird"), CLDataTransform(transform['train'], transform['train_strong_aug']), transform['test']) 
        elif params.dataset == 'web-car':
            dataset = build_webfg_dataset(os.path.join("/data/fgcv/web-data/", "web-car"), CLDataTransform(transform['train'], transform['train_strong_aug']), transform['test']) 
        elif params.dataset == 'web-aircraft':
            dataset = build_webfg_dataset(os.path.join("/data/fgcv/web-data/", "web-aircraft"), CLDataTransform(transform['train'], transform['train_strong_aug']), transform['test'])
        
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)

    return dataset, train_loader, test_loader



def loss_jocor(y_1, y_2, t):
    co_lambda=0.5
    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()

    loss = torch.mean(loss_pick[:])

    return loss


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


if __name__ == '__main__':
    main()
