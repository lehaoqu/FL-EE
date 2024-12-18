set -ex

sr=0.1
total_num=100

rnd=500
md=vit
bs=32
lr=0.05
# cp=models/facebook/deit-small-patch16-224

# python main.py eefl             $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
python main.py darkflpg         $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr
python main.py darkflpa2        $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --kd_lr $lr --s_epoches 1
python main.py depthfl          $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
# python main.py reefl            $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr 0.005 --bs $bs 
python main.py scalefl          $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
python main.py exclusivefl      $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
python main.py inclusivefl      $3  --rnd $rnd --ft $2 --suffix $1/${2}_${3} --device $4 --dataset speechcmds --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs 
