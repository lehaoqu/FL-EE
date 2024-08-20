dts=cifar100-224-d03
md=vit
sr=0.1
total_num=120

python main.py depthfl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
python main.py scalefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
python main.py exclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
python main.py inclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3
# python main.py heterofl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3


# python main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
# python main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr