dts=cifar100-224-d03
md=vit
sr=1

python ../main.py depthfl --suffix $1 --dataset $dts --model $md --sr $sr
# python ../main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
# python ../main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr