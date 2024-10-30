set -ex

cp=models/google-bert/bert-12-128-uncased

# datasets=(sst2 qnli qqp)
datasets=(sst2 qnli qqp)


for ds in "${datasets[@]}"
do
    python main.py ours2 l2w --suffix exps/ours2/$ds --device 3 --dataset $ds --model bert --sr 0.1 --total_num 120 --lr $1 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_alpha 10 --g_eta 5 --config_path $cp --g_gamma 20 
    # python main.py ours2 $1 --suffix exps/ours2/$ds --device 2 --dataset $ds --model bert --sr 0.1 --total_num 120 --lr 0.05 --is_latent True --kd_lr 1e-3 --g_lr 1e-2 --g_alpha 10 --g_eta 0.5 --config_path $cp --g_gamma 100  
done