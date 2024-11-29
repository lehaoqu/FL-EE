set -ex

sr=0.1
total_num=100

md=vit
bs=32
lr=0.05
cp=models/facebook/deit-small-patch16-224

python main.py eefl             $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
python main.py depthfl          $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
python main.py darkflpg         $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
# python main.py reefl            $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr 0.005 --bs $bs --config_path $cp
# python main.py scalefl          $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
# python main.py exclusivefl      $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
# python main.py inclusivefl      $3  --ft $2 --suffix $1/${2}_${3} --device $4 --dataset imagenet --model $md --sr $sr --total_num $total_num --lr $lr --bs $bs --config_path $cp
