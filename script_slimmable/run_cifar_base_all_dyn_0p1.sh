set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05

noniids=(0.1)
# noniids=(1000)

for noniid in "${noniids[@]}"
do
        python main.py darkflpg         $3  --sw static     --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.9 --slim_ce --slim_kd --slim_kd_dyn_weights
        python main.py darkflpa2        $3  --s_gamma 0.99  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.9 --slim_ce --slim_kd --slim_kd_dyn_weights

        python main.py darkflpg         $3  --sw static     --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.95 --slim_ce --slim_kd --slim_kd_dyn_weights
        python main.py darkflpa2        $3  --s_gamma 0.99  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.95 --slim_ce --slim_kd --slim_kd_dyn_weights
        
        python main.py darkflpg         $3  --sw static     --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.95 0.9 --slim_ce --slim_kd --slim_kd_dyn_weights
        python main.py darkflpa2        $3  --s_gamma 0.99  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.95 0.9 --slim_ce --slim_kd --slim_kd_dyn_weights
        
        python main.py darkflpg         $3  --sw static     --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.85 --slim_ce --slim_kd --slim_kd_dyn_weights
        python main.py darkflpa2        $3  --s_gamma 0.99  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios 1.0 0.85 --slim_ce --slim_kd --slim_kd_dyn_weights
done
