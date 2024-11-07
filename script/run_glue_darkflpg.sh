set -ex

md=bert
cp=models/google-bert/bert-12-128-uncased

sr=0.1
total_num=120


bs=32
optim=sgd
lr=5e-2


# datasets=(sst2 qnli qqp)
datasets=(sst2 qnli qqp)

for ds in "${datasets[@]}"
do
    python main.py darkflpg boosted --suffix exps/1107_glue_small/darkflpg/$ds --agg after --kd_direction sl --kd_knowledge relation --kd_join last --device $1 --dataset $ds --model $md --sr 0.1 --total_num 120 --lr 0.05 --s_epoches 2 --valid_gap 10 --rnd 500 --is_feature True --loss_type kd_loss --sw learn --sw_type soft --config_path $cp --eq_ratios 0.75 0.083 0.083 0.083
done

