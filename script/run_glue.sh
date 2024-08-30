sr=1
total_num=12

dts=sst2
md=bert
cp=models/google-bert/bert-base-uncased


python main.py reefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
python main.py depthfl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
python main.py scalefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
python main.py inclusivefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 
python main.py exclusivefl $4 --suffix $1 --device $2 --dataset $dts --model $md --sr $sr --total_num $total_num --lr $3 

