CUDA_VISIBLE_DEVICES=1 python -u Web_fgcv.py \
   --exp-dir experiment/web_bird-sota --dataset web-bird --seed 100  --gpu 0   \
   --arch resnet50 --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
   --epochs 100     --loss_weight1 0.1 --loss_weight2 0.3 --loss_weight4 1.5 --proto_m 0.998   --noise-type symmetric \
   --n_classes 200 --warmupepochs 10   --epsilon 0.6   --momentum 0.8  --moco_m 0.99\
   --topk 15 --forget_rate 0.25  --rechange_per_epoch 35 --conf_ema_range 0.8,0.9 --low-dim 512  --batch-size 48 

