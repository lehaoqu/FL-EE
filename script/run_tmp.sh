sr=1
total_num=4

dts=cifar100-224-d03
md=vit
pl=boosted

python main.py ours1 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --policy $4
# python main.py depthfl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --policy $4
# python main.py scalefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --policy $4
# python main.py inclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --policy $4
# python main.py exclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --policy $4

# python main.py heterofl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3



