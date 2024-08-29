sr=1
total_num=4

dts=cifar100-224-d03
md=vit


python main.py ours2 base --suffix exps/tmp --device 2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $1 
# python main.py reefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
# python main.py depthfl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
# python main.py scalefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
# python main.py inclusivefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
# python main.py reefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 



