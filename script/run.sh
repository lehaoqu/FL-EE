dts=cifar100-224-d03
md=vit
sr=0.1
total_num=120

python ../main.py $1 --suffix $2 --device $3 --dataset $dts --model $md --sr $sr --total_num $total_num --lr 0.01 
# python ../main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
# python ../main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr