set -ex

sr=0.1
total_num=120

# dts=cifar100-224-d03
# dts=cifar100-224-d03-1
md=vit
bs=32
lr=0.05

device=$1
agg=$2
is_feature=$3

kd_directions=(sl ls sls)
kd_knowledges=(response)
kd_joins=(last all)

for kd_direction in "${kd_directions[@]}"
do
    echo "$ply" 
    for kd_knowledge in "${kd_knowledges[@]}"
    do
        for kd_join in "${kd_joins[@]}"
        do
            python main.py darkflpup boosted --suffix exps/try/darkflpup_$agg-$kd_direction-$kd_knowledge-$kd_join-$is_feature/ --agg $agg --kd_direction $kd_direction --kd_knowledge $kd_knowledge --kd_join $kd_join --device $device --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05  --s_epoches 2 --valid_gap 10 --rnd 100 --is_feature $is_feature
        done
        done
    done
    
done