CUDA_VISIBLE_DEVICES=0 python -u Cifar.py \
   --exp-dir experiment/cifar-sota --dataset cifar100 --seed 100  --gpu 0   \
   --arch sevenCNN --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
   --epochs 200  --loss_weight1 1.3 --loss_weight2 0.7  --proto_m 0.988   --noise-type symmetric \
   --closeset_ratio 0.2 --synthetic_data cifar100nc --n_classes 100 --warmupepochs 30   --epsilon 0.6  \
   --topk 10 --forget_rate 0.2  --rechange_per_epoch 30 --conf_ema_range 0.9,0.5 --low-dim 256  
