* * *

Sentiment Analysis on Movie Reviews（Kaggle）学习项目
===============================================

本仓库记录了我从零开始参加 Kaggle 竞赛 **Sentiment Analysis on Movie Reviews** 的完整流程：  
从 **TF-IDF + 线性模型 baseline** 到 **GPU 上微调 RoBERTa（Transformers）**，并最终产出可提交的 `submission.csv`。

*   线性模型最佳 Public Score（示例）：`0.63623`
    
*   RoBERTa GPU 微调 Public Score（示例）：`0.70138`
    

> 你可以把本仓库当作一个"可复现的 NLP 比赛入门模板"。

* * *

## 目录

*   [项目目标](#项目目标)
    
*   [任务简介](#任务简介)
    
*   [仓库结构](#仓库结构)
    
*   [环境与依赖](#环境与依赖)
    
*   [快速开始](#快速开始)
    
*   [Baseline：TF-IDF + 线性模型（可提交）](#baseline-tf-idf--线性模型可提交)
    
*   [GPU 提分：RoBERTa 微调（Transformers，可提交）](#gpu-提分roberta-微调transformers可提交)
    
*   [提交到 Kaggle](#提交到-kaggle)
    
*   [常见问题（Windows / GPU / Transformers）](#常见问题windows--gpu--transformers)
    
*   [进一步改进方向](#进一步改进方向)
    

* * *

## 项目目标

1.  学会 Kaggle NLP 比赛的标准流程：  
    环境搭建 → 读数据 → 划分验证集 → 训练 → 生成提交 → 提分迭代
    
2.  提供两个可复现方案：
    
    *   传统 ML baseline（快、稳、易理解）
        
    *   GPU 上 Transformers 微调（更高分）
        

* * *

## 任务简介

*   输入：电影评论的短语（`Phrase`）
    
*   输出：情感类别（`Sentiment`，5 分类：0~4）
    
*   评估指标：accuracy
    
*   提交格式：`PhraseId,Sentiment`
    

数据文件（下载后放在 `data/`）：

*   `train.tsv`
    
*   `test.tsv`
    
*   `sampleSubmission.csv`
    

* * *

## 仓库结构

    .
    ├─ src/
    │  ├─ baseline.py                  # TF-IDF + LogisticRegression baseline（生成提交）
    │  └─ train_roberta_gpu.py          # RoBERTa 微调（GPU）（生成提交）
    ├─ data/                            # Kaggle 数据（不上传 GitHub）
    ├─ models/                          # 训练输出（不上传 GitHub）
    ├─ submissions/                     # 本地生成的提交文件（建议不上传）
    ├─ requirements.txt                 # 公共依赖（不强绑定 torch/cuXXX）
    ├─ requirements-lock.txt            # 本机完整锁版本（可选）
    ├─ .gitignore
    └─ README.md
    

* * *

## 环境与依赖

*   Windows 11 + PowerShell（本 README 默认 PowerShell）
    
*   Python 3.9+（推荐 3.10/3.11 也可）
    
*   VSCode（可选但推荐）
    
*   GPU 训练（可选）：
    
    *   NVIDIA GPU（如 RTX 3050 Ti）
        
    *   正常的 NVIDIA 驱动
        
    *   PyTorch GPU 版本（本仓库示例：cu124）
        

* * *

## 快速开始

### 1) 克隆仓库

    git clone https://github.com/yinmo318/Sentiment-Analysis-on-Movie-Reviews.git
    cd Sentiment-Analysis-on-Movie-Reviews
    

### 2) 创建并激活虚拟环境（Windows PowerShell）

    python -m venv .venv
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install -U pip wheel
    

### 3) 安装项目依赖

    pip install -r requirements.txt
    

### 4) 下载 Kaggle 数据到 `data/`

本仓库不包含 Kaggle 竞赛数据。请先加入竞赛并接受规则，然后使用 Kaggle API 下载：

    pip install kaggle
    kaggle competitions download -c sentiment-analysis-on-movie-reviews -p data --unzip
    

下载后检查文件是否齐全：

    dir .\\data
    # 应包含 train.tsv / test.tsv / sampleSubmission.csv
    

* * *

## Baseline：TF-IDF + 线性模型（可提交）

### 运行 baseline 并生成提交文件

    python .\\src\\baseline.py
    

脚本会：

*   读取 `data/train.tsv` 和 `data/test.tsv`
    
*   划分验证集并打印验证 accuracy（本项目通常按 `SentenceId` 分组切分，避免泄漏）
    
*   在 `submissions/` 下生成提交文件（示例：`submission_*.csv`）
    

> Baseline 的学习重点
> 
> *   TF-IDF 的 word/char n-gram 特征
>     
> *   线性模型（LogisticRegression）
>     
> *   验证集设计（按 `SentenceId` 分组更可靠）
>     

* * *

## GPU 提分：RoBERTa 微调（Transformers，可提交）

### 1) 安装 PyTorch GPU（示例：cu124）

> 如果你是其他 CUDA 版本，请按自己的情况选择对应的 PyTorch 安装源。

    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    

验证 GPU 是否可用：

    python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
    

### 2) 运行训练脚本并生成提交文件

    python .\\src\\train_roberta_gpu.py
    

脚本会：

*   读取数据并按 `SentenceId` 划分训练/验证
    
*   微调 `roberta-base`
    
*   输出验证集 `eval_accuracy`
    
*   在 `submissions/` 下生成提交文件（示例：`submission_roberta_base_gpu.csv`）
    

### 3) 提速建议（笔记本显卡常用）

如果你觉得训练慢，优先尝试：

*   把 `max_length` 从 96 降到 64（短语任务一般足够）
    
*   在显存允许下：
    
    *   增大 `per_device_train_batch_size`
        
    *   减少 `gradient_accumulation_steps`  
        （通常能明显加速）
        

* * *

## 提交到 Kaggle

1.  进入比赛页面 → Submit Predictions
    
2.  上传你生成的文件（位于 `submissions/`）
    
    *   baseline 输出：`submissions/submission_*.csv`
        
    *   roberta 输出：`submissions/submission_roberta_base_gpu.csv`
        
3.  等待 Kaggle 给出 Public Score
    

提交文件格式检查（可选）：

    Get-Content .\\submissions\\submission_roberta_base_gpu.csv -TotalCount 5
    

应类似：

    PhraseId,Sentiment
    156061,3
    156062,3
    ...
    

* * *

## 常见问题（Windows / GPU / Transformers）

### 1) Tokenizer 报错：`TextEncodeInput must be Union[...]`

通常原因：输入里有 NaN/None/非字符串。  
解决思路：tokenize 前把文本强制转成 `str()` 并处理缺失值。

### 2) `TrainingArguments` 参数不兼容（例如 `evaluation_strategy`）

Transformers 不同版本参数名可能变化。  
例如：有的版本使用 `eval_strategy` 而不是 `evaluation_strategy`。  
遇到报错时以你当前安装版本的提示为准，按报错改参数名即可。

### 3) Windows 下 Hugging Face cache symlink 警告

这是警告，不影响训练。  
如需优化缓存，可开启 Windows "开发者模式"或以管理员运行。

### 4) GitHub 推送网络问题

如果 HTTPS 连接不稳定：

*   尝试强制 HTTP/1.1：`git config --global http.version HTTP/1.1`
    
*   或使用 SSH over 443（适合部分网络环境）
    

* * *

## 进一步改进方向

*   更稳定评估：GroupKFold（按 `SentenceId` 分组的 K 折交叉验证）
    
*   进一步提速：降低 `max_length`、减少梯度累积、开启 `torch.compile`
    
*   更强模型：DeBERTa / RoBERTa-large（注意显存、batch）
    
*   工程化：把训练参数做成命令行参数（如 `--model --lr --epochs --max_length`），并记录实验日志
    
* * *
