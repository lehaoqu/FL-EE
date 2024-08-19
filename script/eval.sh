dts=cifar100-224-d03
md=vit
vr=0.2

python eval.py depthfl --suffix $1 --device $2 --dataset $dts --model $md --valid_ratio $vr --if_mode anytime
# python main.py scalefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
# python main.py exclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
# python main.py inclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
# python main.py heterofl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3


# python main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
# python main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr