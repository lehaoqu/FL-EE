set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05

noniids=(1000 1 0.1)

ns=(2 16 32 64 100)
hds=(64 128 256 512)

hd=$5

for n in "${ns[@]}"
do
        # for hd in "${hds[@]}"
        # do
                python main.py darkflpg         $3 --noise $n --hidden_dim $hd  --sw static  --ft $2 --suffix $1/GENERATOR_ABLATION/${2}_${3}/$hd/$n --device $4 --dataset cifar100_noniid1000 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs
                python main.py darkflpa2        $3 --noise $n --hidden_dim $hd  --s_gamma 0.99  --ft $2 --suffix $1/GENERATOR_ABLATION/${2}_${3}/$hd/$n --device $4 --dataset cifar100_noniid1000 --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
        # done
done
