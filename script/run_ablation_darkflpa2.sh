set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05
declare -a eq_ratios

ply=boosted

# python main.py darkflpa2         $ply                   --ft full --suffix $1/${2}_increase/    --device $3 --dataset cifar100_noniid1000 --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
python main.py darkflpa2         $ply  --increase       --ft full --suffix $1/noniid1000_nonincrease/   --device $2 --dataset cifar100_noniid1000   --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
python main.py darkflpa2         $ply  --increase       --ft full --suffix $1/noniid1_nonincrease/      --device $2 --dataset cifar100_noniid1      --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 
python main.py darkflpa2         $ply  --increase       --ft full --suffix $1/noniid0.1_nonincrease/    --device $2 --dataset cifar100_noniid0.1    --model $md --sr $sr --total_num $total_num --lr $lr       --bs $bs 

python main.py darkflpa2         $ply  --increase       --ft full --suffix $1/speechcmds_nonincrease --device $2 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr --s_epoches 1
python main.py darkflpa2         $ply  --increase       --ft full --suffix $1/svhn_nonincrease/ --device $2 --dataset svhn --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --s_epoches 2