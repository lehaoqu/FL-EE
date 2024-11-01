set -ex

sr=0.1
total_num=120

# dts=cifar100-224-d03
# dts=cifar100-224-d03-1

# change kd_join to [all]:  test kd_join [all] may acc increase
python main.py darkflpgup boosted --suffix exps/darkflpgup_5e-2_1101/$1_soft --kd_join all --device $2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --s_epoches 2 --valid_gap 10 --rnd 500 --is_feature True --loss_type $1 --sw learn --sw_type soft --s_epoches 2

# change gap_kd_lambda to 0.7
python main.py darkflpgup boosted --suffix exps/darkflpgup_5e-2_1101/$1_soft --kd_join last --device $2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --s_epoches 2 --valid_gap 10 --rnd 500 --is_feature True --loss_type $1 --sw learn --sw_type soft --s_epoches 2 --gap_kd_lambda 0.7

# change s_epoches to [2]:  test s_epoches increase may acc increase
python main.py darkflpgup boosted --suffix exps/darkflpgup_5e-2_1101/$1_soft --kd_join last --device $2 --dataset cifar100-224-d03 --model vit --sr 0.1 --total_num 120 --lr 0.05 --s_epoches 2 --valid_gap 10 --rnd 500 --is_feature True --loss_type $1 --sw learn --sw_type soft --s_epoches 10
