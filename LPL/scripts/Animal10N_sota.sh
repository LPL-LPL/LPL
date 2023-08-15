# Animal10N softmax==0.001
CUDA_VISIBLE_DEVICES=1 python -u Animal10N.py \
   --exp-dir experiment/Animal10N-sota --dataset Animal10N --seed 100  --gpu 0   \
   --arch vgg19_bn --moco_queue 8192 --prot_start 1 --lr 0.02 --wd 5e-4  --cosine \
   --epochs 150     --loss_weight1 0.8 --loss_weight2 1.5  --proto_m 0.998 \
    --n_classes 196 --warmupepochs 12   --epsilon 0.53   --momentum 0.99\
   --topk 5 --forget_rate 0.20  --rechange_per_epoch 30 --conf_ema_range 0.8,0.9 --low-dim 512  --batch-size 128 
