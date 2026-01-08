set -ex

sr=0.1
total_num=100

bs=32
lr=0.05
md=bert
cp=models/bert_uncased_L-12_H-128_A-2
g=0.99
datasets=(sst2)
noises=(10 20 50 100)

for ds in "${datasets[@]}"
do
    for n in "${noises[@]}"
    do 
        python main.py darkflpg         $3  --eval_test --noise $n --sw static  --ft $2 --suffix $1/${2}_${3}/$ds/$n/e --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
        python main.py darkflpg         $3              --noise $n --sw static  --ft $2 --suffix $1/${2}_${3}/$ds/$n/n --device $4 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr    --bs $bs --config_path $cp
    done
done
