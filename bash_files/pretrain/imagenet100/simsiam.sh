
listVar="0.3 0.4 0.5 0.6 0.7"
for i in $listVar
do 
    echo "$i"
    python3 ../../../main_pretrain_prun.py \
        --dataset imagenet100 \
        --backbone resnet18 \
        --data_dir ./datasets \
        --train_dir imagenet100/train \
        --val_dir imagenet100/val \
        --max_epochs 500 \
        --gpus 0,1 \
        --accelerator gpu \
        --strategy ddp \
        --sync_batchnorm \
        --precision 16 \
        --optimizer sgd \
        --scheduler warmup_cosine \
        --lr 0.5 \
        --classifier_lr 0.1 \
        --weight_decay 1e-5 \
        --batch_size 128 \
        --num_workers 4 \
        --brightness 0.4 \
        --contrast 0.4 \
        --saturation 0.4 \
        --hue 0.1 \
        --num_crops_per_aug 2 \
        --zero_init_residual \
        --name simsiam \
        --project channel_gating \
        --entity tkrishna \
        --wandb \
        --den-target $i \
        --lbda 5 \
        --gamma 1 \
        --alpha 2e-2 \
        --save_checkpoint \
        --method simsiam \
        --base_proj_hidden_dim 32 \
        --base_pred_hidden_dim 8 \
        --base_proj_output_dim 32 \
        --knn_eval \
        --width 64 \

    sleep 1m
done 