# DSPy + OPRO Haiku 最小测试

这个目录用 `haiku_bot` 做一个最小的 DSPy + OPRO Prompt 优化实验。

核心分工：

- DSPy 定义并执行 `location + season + mood -> haiku` 程序。
- OPRO 外层循环维护 `instruction -> score -> feedback` 历史。
- Optimizer LLM 根据历史生成新的候选 instruction。
- 本地启发式 metric 负责快速评分，避免依赖 spaCy 大模型。

## 项目结构

```text
testDSPyOPRO/
  README.md                    # 实验说明与运行指令
  requirements.txt             # 最小依赖，只需要 DSPy
  haiku_examples.jsonl         # 本实验自带样本数据
  test_dspy_opro_haiku.py      # 主实验脚本：DSPy 程序 + OPRO 循环 + metric
  .gitignore                   # 忽略 key、缓存和运行产物
  prompts/
    best_haiku_instruction.txt # 运行时自动创建：当前最佳 instruction
  runs/
    haiku_opro_run_*.json      # 运行时自动创建：结构化完整记录
    haiku_opro_report_*.md     # 运行时自动创建：可读实验报告
```

默认数据集使用同目录文件：

```text
haiku_examples.jsonl
```

默认 API key 读取顺序：

1. 环境变量 `DEEPSEEK_API_KEY`
2. `testDSPyOPRO/apikey.txt`

## Metric 评分标准

本实验使用轻量启发式 metric，不依赖 spaCy，目标是快速验证 OPRO 流程是否有效。总分范围是 `0~1`，主要看：

- 三行结构：是否输出 3 行 haiku。
- 音节形状：是否接近 `5-7-5`。
- 季节表达：避免直接复读输入季节，用具体意象暗示。
- 地点锚定：是否包含或呼应输入 location 的具体细节。
- 情绪表达：是否通过画面、声音、动作间接呈现 mood。
- 输出纯净度：不要标题、解释或多余说明。

这个 metric 不是文学质量裁判，更像一个便宜的自动验收器；后续如果要更严肃评估，可以替换成基于 NLP 工具或人工标注集的更细评分器。

## OPRO 优化策略

每轮优化流程很简单：

1. 先评估几条 seed instruction。
2. 把历史 `instruction + score + feedback` 放进 optimizer prompt。
3. 让 LLM 生成新的候选 instruction。
4. 用同一个 DSPy `HaikuBot` 跑训练样本并打分。
5. 记录结果，继续下一轮，最后选择训练集最高分 prompt。
6. 对最佳 prompt 再跑 val/test，并保存 JSON、Markdown 报告和 best prompt。

这里 DSPy 负责稳定执行任务，OPRO 只负责外层搜索 prompt。后续可以直接加载 `prompts/best_haiku_instruction.txt` 做固定 Prompt 推理，也可以把它作为下一轮 OPRO 的初始 seed 继续优化。

## 克隆与安装

从 GitHub 克隆项目后进入本目录：

```bash
git clone https://github.com/KujoStar/testDSPywithOPRO.git
cd testDSPywithOPRO
```

建议在已有 DSPy 环境中运行。如果要单独安装：

```bash
python -m pip install -r requirements.txt
```

## 配置 API Key

脚本默认使用 `deepseek/deepseek-v4-flash`，也可以用其他 LiteLLM 支持的模型。均需要用户自己配置 LLM API key。推荐使用环境变量：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

也可以在 `testDSPyOPRO` 目录下新建本地文件：

```bash
printf "your_api_key_here" > apikey.txt
```

`apikey.txt` 已经在 `.gitignore` 中，不要把自己的 API key 提交到 GitHub。

## 最小运行

```bash
python test_dspy_opro_haiku.py
```

脚本默认只跑小预算：

```text
rounds=2
candidates_per_round=2
train_size=6
val_size=3
test_size=3
```

如果 `prompts/best_haiku_instruction.txt` 已经存在，脚本会在运行开始时自动把它加入 seed instructions，并优先参与本次 OPRO 搜索。运行过程中不会实时覆盖该文件；每次脚本完整运行结束后，才会用本次 `train_score` 最高的 instruction 更新它。

## 更小预算 Smoke Test

```bash
python test_dspy_opro_haiku.py \
  --rounds 1 \
  --candidates-per-round 1 \
  --train-size 3 \
  --val-size 1 \
  --test-size 1
```

## 扩大搜索预算

```bash
python test_dspy_opro_haiku.py \
  --rounds 4 \
  --candidates-per-round 3 \
  --train-size 12 \
  --val-size 4 \
  --test-size 4
```

## 输出

运行结束后会打印：

- best instruction
- train / val / test score
- run JSON 保存路径，适合程序读取
- run Markdown 报告保存路径，适合人工阅读和长期归档
- best prompt 保存路径

其中：

```text
prompts/best_haiku_instruction.txt
runs/haiku_opro_run_*.json
runs/haiku_opro_report_*.md
```

`prompts/best_haiku_instruction.txt` 可以作为后续固定 Prompt 推理入口。
