# set -ex

# sr=0.1
# total_num=100

# rnd=500
# md=vit
# bs=32
# lr=0.05
# # cp=models/facebook/deit-small-patch16-224

# # python main.py eefl             $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 

# ses=(1 2 5 10)
# sgs=(1 0.99)
# for s in "${ses[@]}" 
# do
#     for sg in "${sgs[@]}"
#     do
#         python main.py darkflpg         $2  --s_epoches $s  --s_gamma $sg   --rnd $rnd --ft lora --suffix $1/lora_${2}/$sg/$s --device $3 --dataset speechcmds    --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr
#     done
# done



set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05
declare -a eq_ratios
noniids=(1000 1 0.1)


if [ "$3" == "small" ]; then
        eq_ratios="0.4 0.3 0.2 0.1"
fi

if [ "$3" == "normal" ]; then
        eq_ratios="0.25 0.25 0.25 0.25"
fi

if [ "$3" == "large" ]; then
        eq_ratios="0.1 0.2 0.3 0.4"
fi

ply=$5

if [ "$2" == "lora" ]; then
    if [ "$5" == "base" ]; then
        lr=0.01
    fi
fi

for noniid in "${noniids[@]}"
do
        python main.py scalefl          $ply   --ft $2 --suffix $1/${2}_${3}_{$ply}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --eq_ratios $eq_ratios
done