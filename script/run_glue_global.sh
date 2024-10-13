set -ex

md=bert
cp=models/google-bert/bert-12-uncased

sr=0.1
total_num=120


bs=32
optim=sgd
lr=5e-2
epoch=1


datasets=(sst2 qqp qnli)
ply=(base boosted l2w)

for ds in "${datasets[@]}"
do
    echo "$ply" 
    python global.py depthfl      $ply --suffix $1/$ds --device $2 --dataset $ds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp --optim $optim --epoch $epoch
done
