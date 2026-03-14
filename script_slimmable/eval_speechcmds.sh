set -ex

# dts=cifar100-224-d03
# md=vit

vr=0.2

fts=(full lora)
pls=(small large)
# noniids=(1000)
noniids=(1 0.1)

ft=$4
pl=$5

# for noniid in "${noniids[@]}"
# do
#     python eval_slimmable.py eefl boosted --suffix $1/${ft}_$pl/noniid$noniid --device $2 --dataset cifar100_noniid$noniid --model $3 --valid_ratio $vr --if_mode all --ft $ft
# done

python eval_slimmable.py eefl boosted --suffix $1/${ft}_$pl --device $2 --dataset speechcmds --model $3 --valid_ratio $vr --if_mode all --ft $ft
