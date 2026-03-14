import os
import json
import re
import pandas as pd
from openpyxl.styles import Font

def process_eval_files(target_dir, origin_dir, output_name):
    # 指定要读取的 exits config
    target_exits_configs = ["[2, 5, 8, 11]", "[2, 5, 8]", "[2, 5]"]
    
    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{output_name}.xlsx")
    
    with pd.ExcelWriter(output_path) as writer:
        # 最外层循环：根据 exits config 遍历
        for exits_config in target_exits_configs:
            rows = []
            
            # 1. 读取 Slimmable 目录下的相关文件
            if os.path.exists(target_dir):
                file_names = os.listdir(target_dir)
                # 匹配模式: base_slim_ratio_exits_[...]_eval.json
                slimm_pattern = re.compile(rf'^(?P<base>.*)_slim_(?P<ratio>[\d\.]+)_exits_{re.escape(exits_config)}_eval\.json$')
                
                for fname in file_names:
                    match = slimm_pattern.match(fname)
                    if match:
                        base_name = match.group('base')
                        alg_name = base_name.split('_')[0]
                        slim_ratios_config = base_name.split('_')[-1]
                        
                        # 排序 slim_ratios_config 中的比率，例如 [1.0-0.9-0.8] -> [0.8-0.9-1.0]
                        try:
                            inner = slim_ratios_config.strip('[]')
                            if '-' in inner:
                                parts = sorted(inner.split('-'), key=float)
                                slim_ratios_config = f"[{'-'.join(parts)}]"
                        except Exception:
                            pass
                        
                        # print(fname, slim_ratios_config)
                        
                        
                        need = ["[0.8-0.9-1.0]","[0.8-1.0]","[0.85-1.0]","[0.9-1.0]","[0.9-0.95-1.0]"]
                        if slim_ratios_config not in need:
                            continue

                        ratio = match.group('ratio')
                        
                        # 按之前逻辑，只处理 ratio 为 1.0 的
                        if ratio != '1.0':
                            continue
                            
                        file_path = os.path.join(target_dir, fname)
                        # print(file_path)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            rows.append({
                                'Algorithm': alg_name,
                                'Slim Ratios': slim_ratios_config,
                                'All Slim Budgeted AUC': data.get('all_slim_budgeted_auc'),
                                'Min FLOPs': data.get('min_flops'),
                                'Max FLOPs': data.get('max_flops'),
                            })
                        except Exception as e:
                            print(f"Error reading slimm file {fname}: {e}")

            # 2. 读取 Origin 目录下的相关文件
            if os.path.exists(origin_dir):
                origin_files = os.listdir(origin_dir)
                # 匹配模式: (alg)_..._exits_[...]_eval.json
                origin_pattern = re.compile(rf'^(?P<alg>[^_]+)_.*_exits_{re.escape(exits_config)}_eval\.json$')
                
                for origin_fname in origin_files:
                    match = origin_pattern.match(origin_fname)
                    if match:
                        alg = match.group('alg')
                        if 'darkfl' not in alg:
                            continue
                        
                        file_path = os.path.join(origin_dir, origin_fname)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            rows.append({
                                'Algorithm': alg,
                                'Slim Ratios': "BASE",
                                'All Slim Budgeted AUC': data.get('budgeted_auc'),
                                'Min FLOPs': data.get('flops')[0],
                                'Max FLOPs': data.get('flops')[-1],
                            })
                        except Exception as e:
                            print(f"Error reading origin file {origin_fname}: {e}")

            # 3. 如果该 exits_config 下有数据，计算统计值并写入 Sheet
            if rows:
                df = pd.DataFrame(rows)
                
                # 统一计算当前 Sheet 的全局 Min/Max FLOPs
                global_min_flops = df['Min FLOPs'].min()
                global_max_flops = df['Max FLOPs'].max()
                global_d = global_max_flops - global_min_flops
                
                if global_d > 0:
                    df['Global Avg Acc'] = df['All Slim Budgeted AUC'] / global_d
                else:
                    df['Global Avg Acc'] = 0
                
                # 基准核对列
                df['Config Min FLOPs'] = global_min_flops
                df['Config Max FLOPs'] = global_max_flops
                df['Config D'] = global_d

                # 排序与列调整
                cols = ['Algorithm', 'Slim Ratios', 'All Slim Budgeted AUC', 
                        'Min FLOPs', 'Max FLOPs', 'Global Avg Acc', 'Config Min FLOPs', 'Config Max FLOPs', 'Config D']
                df = df[[c for c in cols if c in df.columns]]
                
                # 主关键字 Algorithm, 次关键字 Slim Ratios，均为降序
                df = df.sort_values(by=['Algorithm', 'Slim Ratios'], ascending=[False, False])
                df = df.reset_index(drop=True)  # 重要：重置索引以保证 Excel 写入行号和 DataFrame 索引一致
                
                # 写入 Sheet
                sheet_name = f"Exits_{exits_config.replace('[', '').replace(']', '').replace(' ', '').replace(',', '_')}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 获取 openpyxl 工作表对象进行样式处理
                worksheet = writer.sheets[sheet_name]
                
                # 找到 Global Avg Acc 所在的列索引 (1-based)
                try:
                    global_avg_col_idx = df.columns.get_loc('Global Avg Acc') + 1
                    alg_col_idx = df.columns.get_loc('Algorithm') + 1
                except KeyError:
                    continue

                # 在每个 Algorithm 分组中寻找最好和次好的 Global Avg Acc
                # 遍历 Algorithm 的唯一值
                for alg in df['Algorithm'].unique():
                    # 获取该 Algorithm 对应的行索引（df 的索引 + 2，因为 Excel 从 1 开始且有标题行）
                    alg_indices = df[df['Algorithm'] == alg].index
                    
                    # 获取该组的 Global Avg Acc 值及对应的行号
                    # 注意：如果有多行值相同，这里处理前两个
                    group_data = df.loc[alg_indices, ['Global Avg Acc']].copy()
                    group_data['row_num'] = group_data.index + 2
                    
                    # 按 Global Avg Acc 降序排列
                    sorted_group = group_data.sort_values(by='Global Avg Acc', ascending=False)
                    
                    if len(sorted_group) >= 1:
                        # 最好的：加粗
                        best_row = int(sorted_group.iloc[0]['row_num'])
                        cell = worksheet.cell(row=best_row, column=global_avg_col_idx)
                        cell.font = Font(bold=True)
                        
                    if len(sorted_group) >= 2:
                        # 次好的：下划线
                        second_best_row = int(sorted_group.iloc[1]['row_num'])
                        cell = worksheet.cell(row=second_best_row, column=global_avg_col_idx)
                        cell.font = Font(underline='single')

                print(f"Processed exits config: {exits_config}, global_d: {global_d:.4f}")

    print(f"Excel file saved to: {output_path}")

if __name__ == "__main__":

    dirs = {
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR/full_boosted/noniid1000", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000", "BASE_CIFAR_noniid1000"),
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR/full_boosted/noniid1", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1", "BASE_CIFAR_noniid1"),
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR/full_boosted/noniid0.1", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid0.1", "BASE_CIFAR_noniid0.1"),
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_SPEECHCMDS/full_boosted", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_SPEECHCMDS_ORIGIN/full_boosted", "BASE_SPEECHCMDS"),
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ABLATION/full_boosted/noniid1000", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000", "BASE_CIFAR_ABLATION_noniid1000"),
        ("/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR/lora_boosted/noniid1000", "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/lora_boosted/noniid1000", "BASE_CIFAR_noniid1000_lora"),
    }

    for target_dir, origin_dir, output_name in dirs:
        print(f"\nProcessing {target_dir}...")
        process_eval_files(target_dir, origin_dir, output_name)
