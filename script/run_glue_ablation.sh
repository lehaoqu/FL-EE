set -ex

sr=0.1
total_num=100

md=bert
bs=32
cp=models/bert_uncased_L-12_H-128_A-2
lr=0.05
declare -a eq_ratios

ply=boosted

# python main.py darkflpg         $ply                   --sw $2    --ft full --suffix $1/full/${2}_diffg/    --device $3 --dataset svhn --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 


python main.py darkflpg         $ply  --diff_generator --sw $2   --ft full --suffix $1/full/${2}_noneg/    --device $3 --dataset $4 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs  --config_path $cp 

# python main.py darkflpg         $ply                   --sw $2    --ft full --suffix $1/full/noniid0.1/${2}_diffg/    --device $3 --dataset cifar100_noniid0.1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
# python main.py darkflpg         $ply  --diff_generator --sw $2    --ft full --suffix $1/full/noniid0.1/${2}_noneg/    --device $3 --dataset cifar100_noniid0.1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 


# python main.py darkflpg         $ply                   --sw $2    --ft lora --suffix $1/lora/noniid1/${2}_diffg/    --device $3 --dataset cifar100_noniid1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
# python main.py darkflpg         $ply  --diff_generator --sw $2    --ft lora --suffix $1/lora/noniid1/${2}_noneg/    --device $3 --dataset cifar100_noniid1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 

# python main.py darkflpg         $ply                   --sw $2    --ft lora --suffix $1/lora/noniid0.1/${2}_diffg/    --device $3 --dataset cifar100_noniid0.1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
# python main.py darkflpg         $ply  --diff_generator --sw $2    --ft lora --suffix $1/lora/noniid0.1/${2}_noneg/    --device $3 --dataset cifar100_noniid0.1 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
