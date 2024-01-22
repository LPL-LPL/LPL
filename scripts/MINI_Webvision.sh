CUDA_VISIBLE_DEVICES=0 python -u MINI_Webvision.py \
   --exp-dir experiment/mini_webvision-sota --dataset Webvision --seed 100  --gpu 0   \
   --arch InceptionResNetV2 --moco_queue 4096 --prot_start 1 --lr 0.02 --wd 5e-4  --cosine \
   --epochs 200     --loss_weight1 0.4 --loss_weight2 1.0   --loss_weight4 1.0  --proto_m 0.99   --noise-type symmetric \
   --n_classes 50 --warmupepochs 39   --epsilon 0.6  \
   --topk 25 --forget_rate 0.15  --rechange_per_epoch 30 \
   --conf_ema_range 0.4,0.4 --low-dim 512  --batch-size 32  --momentum 0.9 
