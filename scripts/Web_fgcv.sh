# bird softmax==0.001
CUDA_VISIBLE_DEVICES=1 python -u Web_fgcv.py \
   --exp-dir experiment/web_bird-sota --dataset web-bird --seed 100  --gpu 0   \
   --arch resnet50 --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
   --epochs 100     --loss_weight1 0.1 --loss_weight2 0.3 --loss_weight4 1.5 --proto_m 0.998   --noise-type symmetric \
   --n_classes 200 --warmupepochs 10   --epsilon 0.6   --momentum 0.8  --moco_m 0.99\
   --topk 15 --forget_rate 0.25  --rechange_per_epoch 35 --conf_ema_range 0.8,0.9 --low-dim 512  --batch-size 48 

# aircraft softmax==0.001
CUDA_VISIBLE_DEVICES=1 python -u Web_fgcv.py \
   --exp-dir experiment/web_aircraft-sota --dataset web-aircraft --seed 100  --gpu 0   \
   --arch resnet50 --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 11e-4  --cosine \
   --epochs 100     --loss_weight1 0.15 --loss_weight2 0.2  --proto_m 0.998   --noise-type symmetric \
    --n_classes 100 --warmupepochs 10   --epsilon 0.6   --momentum 0.85\
   --topk 5 --forget_rate 0.20  --rechange_per_epoch 30 --conf_ema_range 0.8,0.9 --low-dim 512  --batch-size 48 

# car softmax==0.001
CUDA_VISIBLE_DEVICES=1 python -u Web_fgcv.py \
   --exp-dir experiment/web_car-sota --dataset web-car --seed 100  --gpu 0   \
   --arch resnet50 --moco_queue 4096 --prot_start 1 --lr 0.015 --wd 11e-4  --cosine \
   --epochs 100     --loss_weight1 0.3 --loss_weight2 1.0  --proto_m 0.998   --noise-type symmetric \
    --n_classes 196 --warmupepochs 10   --epsilon 0.53   --momentum 0.9\
   --topk 100 --forget_rate 0.20  --rechange_per_epoch 40 --conf_ema_range 0.9,0.6 --low-dim 512  --batch-size 48 
