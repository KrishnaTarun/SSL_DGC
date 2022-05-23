
# !/bin/bash
# listVar="1.0 0.75 0.5 0.35 0.25"
listVar="0.3 0.4 0.5"
for i in $listVar 
# for i in {1..20..1} 
do 
    echo "$i"

    python ../../../main_knn.py \
    --dataset imagenet100 \
    --data_dir /home/tarun/Documents/PhD/SSL_DGC/bash_files/pretrain/imagenet100/datasets/ \
    --pretrained_checkpoint_dir /home/tarun/Documents/PhD/SSL_DGC/bash_files/pretrain/imagenet100/imagenet100/trained_models/simsiam/den_target/$i \
    --train_dir imagenet100/train \
    --val_dir imagenet100/val \
    --feature_type backbone \
    --batch_size 256 \
    --distance_function euclidean \
    --k 1

    # sleep 2m

done
