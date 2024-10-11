dts=cifar100-224-d03
md=vit

# dts=sst2
# md=bert

vr=0.2

python eval.py depthfl l2w --suffix $1 --device $2 --dataset $dts --model $md --valid_ratio $vr --if_mode anytime --cosine