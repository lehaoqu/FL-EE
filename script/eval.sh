# dts=cifar100-224-d03
# md=vit

vr=0.2

fts=(full lora)
pls=(boosted)
noniids=(1000)

# for ft in "${fts[@]}" 
# do
#     for pl in "${pls[@]}"
#     do
#         for noniid in "${noniids[@]}"
#         do
#             python eval.py eefl boosted --suffix $1/${ft}_$pl/noniid$noniid --device $2 --dataset $3 --model $4 --valid_ratio $vr --if_mode all --ft $ft
#         done
#     done
# done    

for ft in "${fts[@]}" 
do
    for pl in "${pls[@]}"
    do

        python eval.py eefl boosted --suffix $1/${ft}_$pl/noniid --device $2 --dataset $3 --model $4 --valid_ratio $vr --if_mode all --ft $ft

    done
done    