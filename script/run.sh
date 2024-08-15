dts=cifar100-224-d03
md=vit
sr=0.5

python ../main.py heterofl --suffix $1 --dataset $dts --model $md --sr $sr
# python ../main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
# python ../main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr