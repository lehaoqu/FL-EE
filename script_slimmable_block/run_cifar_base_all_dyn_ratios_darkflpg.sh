set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05
slim_ratios=${5:-"1.0 0.8"}
t_model_path='/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
t_config_path='/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'


noniids=(1000 1 0.1)
# noniids=(1000)

for noniid in "${noniids[@]}"
do
        t_model_path=/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid${noniid}/darkflpg_cifar100_noniid${noniid}_vit_100c_1E_lrsgd0.05_boosted.pth
        t_config_path=/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid${noniid}/darkflpg_cifar100_noniid${noniid}_vit_100c_1E_lrsgd0.05_boosted.json
        
        
        python main.py darkflpg         $3  --sw static     --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd --slim_block_wise --slim_logits $6 --slim_features $7 --t_model_path $t_model_path --t_config_path $t_config_path
        # python main.py darkflpa2        $3  --s_gamma 0.99  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd --slim_block_wise --slim_logits $6 --slim_features $7
        # python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios 1.0
        # python main.py depthfl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd
        # python main.py scalefl          $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd
        # python main.py inclusivefl      $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd
        # python main.py exclusivefl      $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd
        # python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3}/noniid$noniid --device $4 --dataset cifar100_noniid$noniid --model $md --sr $sr --total_num $total_num --lr 0.005     --bs $bs --slimmable --slim_ratios $slim_ratios --slim_ce --slim_kd
done
