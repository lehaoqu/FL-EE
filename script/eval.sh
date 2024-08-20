dts=cifar100-224-d03
md=vit
vr=0.2

python eval.py depthfl --suffix $1 --device $2 --dataset $dts --model $md --valid_ratio $vr --if_mode all