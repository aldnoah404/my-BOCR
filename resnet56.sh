#experiment script for mobilenet v1
if [ $1 = "non" ]; then  
    echo "prune model by naive BO
    "
    python bo_static.py \
        --job=train \
        --model=resnet56 \
        --dataset=cifar10 \
        --preserve_ratio=0.5 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset \
        --ckpt_path=./checkpoints/resnet56.th \
        --acc_metric=acc1 \
        --seed=1 \
        --gpu_idx="0" \

elif [ $1 = "static" ]; then  
    echo "prune model by BO with cluster
    "
    python bo_static.py \
        --job=export \
        --export_path=./purned_model/resnet56_static.pth \
        --ratios="1.0, 0.5, 0.5625, 0.5625, 0.4375, 0.5625, 0.5, 0.4375, 0.5, 0.5625, 0.53125, 0.4375, 0.4375, 0.53125, 0.53125, 0.4375, 0.4375, 0.53125, 0.4375, 0.515625, 0.453125, 0.515625, 0.546875, 0.546875, 0.453125, 0.453125, 0.546875, 0.5, 1.0" \
        --model=resnet56 \
        --dataset=cifar10 \
        --preserve_ratio=0.5 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset \
        --ckpt_path=./checkpoints/resnet56.th \
        --acc_metric=acc1 \
        --seed=1 \
        --static_cluster\
        --simlarity="EU"\
        --n_clusters=3\
        --gpu_idx="0" \

elif [ $1 = "db" ]; then  
    echo "prune model by BO direct rollback
    "
    python bo_back.py \
        --job=train \
        --model=resnet56 \
        --dataset=cifar10 \
        --preserve_ratio=0.5 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset \
        --ckpt_path=./checkpoints/resnet56.th \
        --acc_metric=acc1 \
        --seed=1 \
        --static_cluster\
        --simlarity="EU"\
        --n_clusters=3\
        --gpu_idx="0" \


elif [ $1 = "gb" ]; then  
    echo "prune model by BO gradual rollback
    "
    python bo_back_gradual.py \
        --job=train \
        --model=resnet56 \
        --dataset=cifar10 \
        --preserve_ratio=0.5 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset \
        --ckpt_path=./checkpoints/resnet56.th \
        --acc_metric=acc1 \
        --seed=1 \
        --static_cluster\
        --simlarity="EU"\
        --n_clusters=3\
        --bridge_stage=9 \
        --gpu_idx="0" \

else
    echo "
    instruction not recogized!
    "
fi