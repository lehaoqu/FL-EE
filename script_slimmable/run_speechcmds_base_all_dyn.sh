set -ex

sr=0.1
total_num=100

rnd=500
md=vit
bs=32
lr=0.05
slim_ratios=${5:-"1.0 0.8"}
# cp=models/facebook/deit-small-patch16-224

python main.py darkflpg         $3  --sw static --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr     --slimmable --slim_ce --slim_kd --slim_ratios $slim_ratios        --slim_kd_dyn_weights
python main.py darkflpa2        $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr --s_epoches 1   --slimmable --slim_ce --slim_kd --slim_ratios $slim_ratios        --slim_kd_dyn_weights