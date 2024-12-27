import os, re, statistics

suffix = 'exps/BASE_SVHN/lora_base/noniid'
eval_dir = suffix
file_names = os.listdir(eval_dir)
model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]

data = {}    
for model_path in model_paths:
    if '.txt' in model_path:
        # print(model_path)
        base_name = os.path.basename(model_path)
        name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
        print(name_without_extension)

        with open(model_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        
        for line in lines:
            line = line.strip()
            if 'avg' in line:
                l = line
                numbers = re.findall(r'\d+\.\d+', line)[:-1]
                # 将提取的字符串转换为浮点数，并存储在列表中
                data_list = [float(num) for num in numbers]
                data[name_without_extension] = data_list
        dev = statistics.stdev(data_list)
        print(l)
        print(dev)