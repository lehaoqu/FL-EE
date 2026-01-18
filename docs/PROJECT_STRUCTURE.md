# FL-EE 关键目录结构（可视化导览）

下面给出 FL-EE 仓库中最常用、最关键的目录/文件速览（偏“怎么用/去哪改”视角）。

## 1) 一句话定位：从哪里开始看

- 训练/仿真入口：`main.py`
- 评估入口：`eval.py`、`eval_slimmable.py`
- 数据集划分/生成：`generate_*.py` 与 `dataset/`
- 算法实现：`trainer/alg/`
- 通用参数与工具：`utils/options.py` 与 `utils/`
- 一键运行脚本：`script/`
- 可视化 GUI：`gui/front.py`

## 2) 关键目录树（精简版）

> 注：这是“关键路径”精简树，像 `exps/`、`wandb/`、`imgs/` 这类输出目录通常很大，这里只保留入口层级。

```
FL-EE/
├─ main.py                         # 主入口：训练/联邦仿真
├─ eval.py                         # 评估入口
├─ eval_slimmable.py               # slimmable 相关评估
├─ global.py                       # 全局配置/全局变量（项目级别）
├─ requirements.txt                # Python 依赖
│
├─ dataset/                        # 数据集与划分
│  ├─ *_dataset.py                 # 各数据集 Dataset/加载逻辑
│  ├─ utils/                       # 划分与数据处理工具
│  └─ <dataset_name>/              # 生成后的划分结果（如 sst2_noniid1 等）
│
├─ trainer/                        # 核心训练与联邦流程
│  ├─ base.py                      # 同步/基础 Client/Server
│  ├─ asyncbase*                   # 异步/队列仿真基础（如存在）
│  ├─ alg/                         # 各算法实现（EEF L / DarkFL* / ScaleFL 等）
│  ├─ generator/                   # 生成器/适配器相关组件
│  └─ policy/                      # 策略（调度、选择等）
│
├─ utils/                          # 通用工具
│  ├─ options.py                   # 通用参数解析（命令行 args）
│  ├─ train_utils.py               # 训练通用函数
│  ├─ dataprocess.py               # 数据处理
│  └─ modelload/                   # 模型加载封装
│
├─ script/                         # 常用运行脚本（bash）
├─ script_slimmable/               # slimmable 相关脚本
│
├─ gui/                            # Streamlit 图形界面
│  ├─ front.py                     # GUI 入口
│  └─ modules/                     # GUI 页面模块
│
├─ models/                         # 预训练/下载模型缓存（facebook/google 等）
├─ imgs/                           # 画图输出/可视化图片
├─ exps/                           # 实验配置与输出（大量）
├─ front-exps/                     # GUI/前端相关实验输出
├─ EXPS/ EXPS2/                    # 其他实验批次/备份
└─ wandb/                          # W&B 日志
```

## 3) “我想改/我想跑”该看哪里

- 新增/修改联邦算法：`trainer/alg/`（同时参考 `trainer/base.py` / `trainer/asyncbase*`）
- 改超参默认值或加新参数：`utils/options.py`
- 生成数据划分：项目根目录下 `generate_*.py`，以及 `dataset/` 内对应数据集脚本
- 批量跑实验：`script/`（里面是常用组合的可复现命令）
- 想用 GUI 交互：`gui/front.py`

