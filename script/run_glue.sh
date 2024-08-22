sr=1
total_num=12

dts=sst2
md=bert
cp=models/google-bert/bert-base-uncased


python main.py depthfl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --config_path $cp
python main.py scalefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --config_path $cp
python main.py inclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --config_path $cp
python main.py exclusivefl --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 --config_path $cp