# sota代码 不同噪声率下 softmax==0.01

# sym--0.2--------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
   --epochs 200  --loss_weight1 1.3 --loss_weight2 0.7  --proto_m 0.988   --noise-type symmetric \
   --closeset_ratio 0.2 --synthetic_data cifar100nc --n_classes 100 --warmupepochs 30   --epsilon 0.6  \
   --topk 10 --forget_rate 0.2  --rechange_per_epoch 30 --conf_ema_range 0.9,0.5 --low-dim 256  

CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
    --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
    --epochs 200  --loss_weight1 1.3 --loss_weight2 0.7  --proto_m 0.988   --noise-type symmetric \
    --closeset_ratio 0.2 --synthetic_data cifar80no --n_classes 100 --warmupepochs 30   --epsilon 0.6  \
    --topk 10 --forget_rate 0.2  --rechange_per_epoch 30 --conf_ema_range 0.9,0.5 --low-dim 256  

# sym--0.5--------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
    --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
    --epochs 200  --loss_weight1 1.35 --loss_weight2 0.7  --proto_m 0.988   --noise-type symmetric \
    --closeset_ratio 0.5 --synthetic_data cifar100nc --n_classes 100 --warmupepochs 40   --epsilon 0.6  \
    --topk 5 --forget_rate 0.5  --rechange_per_epoch 35 --conf_ema_range 0.9,0.62 --low-dim 512  
    
CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
    --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
    --epochs 200  --loss_weight1 1.35 --loss_weight2 0.7  --proto_m 0.988   --noise-type symmetric \
    --closeset_ratio 0.5 --synthetic_data cifar80no --n_classes 100 --warmupepochs 40   --epsilon 0.6  \
    --topk 5 --forget_rate 0.5  --rechange_per_epoch 35 --conf_ema_range 0.9,0.62 --low-dim 512 

# sym--0.8--------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.03 --wd 5e-4  --cosine \
   --epochs 200  --loss_weight1 1.2 --loss_weight2 0.4  --proto_m 0.988   --noise-type symmetric \
   --closeset_ratio 0.8 --synthetic_data cifar100nc --n_classes 100 --warmupepochs 80   --epsilon 0.6  \
   --topk 20 --forget_rate 0.8  --rechange_per_epoch 45 --conf_ema_range 0.8,0.9 --low-dim 256 

CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.03 --wd 5e-4  --cosine \
   --epochs 200  --loss_weight1 1.2 --loss_weight2 0.4  --proto_m 0.988   --noise-type symmetric \
   --closeset_ratio 0.8 --synthetic_data cifar80no --n_classes 100 --warmupepochs 80   --epsilon 0.6  \
   --topk 20 --forget_rate 0.8  --rechange_per_epoch 45 --conf_ema_range 0.8,0.9 --low-dim 256  

# asym--0.4------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.020 --wd 11e-4  --cosine \
   --epochs 200     --loss_weight1 1.3 --loss_weight2 0.8  --proto_m 0.998   --noise-type asymmetric \
   --closeset_ratio 0.4 --synthetic_data cifar100nc --n_classes 100 --warmupepochs 100   --epsilon 0.6  \
   --topk 3 --forget_rate 0.4  --rechange_per_epoch 35 --conf_ema_range 0.8,0.9 --low-dim 256  

CUDA_VISIBLE_DEVICES=1 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 333  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.020 --wd 11e-4  --cosine \
   --epochs 200     --loss_weight1 1.3 --loss_weight2 0.8  --proto_m 0.998   --noise-type asymmetric \
   --closeset_ratio 0.4 --synthetic_data cifar80no --n_classes 100 --warmupepochs 100   --epsilon 0.6  \
   --topk 5 --forget_rate 0.4  --rechange_per_epoch 35 --conf_ema_range 0.8,0.9 --low-dim 256  
