# dts=cifar100-224-d03
# md=vit

vr=0.2

python eval.py eefl base --suffix $1 --device $2 --dataset $3 --model $4 --valid_ratio $vr --if_mode all
python eval.py eefl boosted --suffix $1 --device $2 --dataset $3 --model $4 --valid_ratio $vr --if_mode all
python eval.py eefl l2w --suffix $1 --device $2 --dataset $3 --model $4 --valid_ratio $vr --if_mode all