# 数据集和评估指标

提供的微调脚本允许您通过将 `dataset` 参数传递给 `llama_recipes.finetuning` 模块或 `examples/finetuning.py` 脚本来在三个数据集之间进行选择。当前的选项是 `grammar_dataset`、`alpaca_dataset` 和 `samsum_dataset`。此外，我们集成了 OpenAssistant/oasst1 数据集作为[自定义数据集的示例](../examples/custom_dataset.py)。注意：对这些数据集的使用应符合数据集的基础许可证（包括但不限于非商业用途）。

* [grammar_dataset](https://huggingface.co/datasets/jfleg) 包含 15 万对英语句子和可能的更正。
* [alpaca_dataset](https://github.com/tatsu-lab/stanford_alpaca) 提供了由 `text-davinci-003` 生成的 5.2 万个指令-响应对。
* [samsum_dataset](https://huggingface.co/datasets/samsum) 包含约 1.6 万个类似于 Messenger 的对话摘要。
* [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1/) 包含约 8.8 万条助手式对话消息。

## 批处理策略
Llama-recipes 支持两种批处理请求的策略。
默认设置为 `packing`，它将标记的样本连接成长序列，填满模型的上下文长度。这是最有效的计算变体，因为它避免了任何填充，所有序列的长度相同。在上下文长度的边界处截断的样本会被截断，切割序列的其余部分用作下一个长序列的开头。

如果训练数据量较小，这个过程可能会引入很多噪音到训练数据中，这可能会损害微调模型的预测性能。因此，我们还支持 `padding` 策略，它不会由于截断序列而引入额外的噪音。该策略试图通过将相似长度的样本批处理在一起来最小化效率损失，因此只需进行最小的填充。

批处理策略可以通过命令行参数 `--batching_strategy [packing]/[padding]` 选择。

## 使用自定义数据集

llama-recipes 中提供的可用数据集列表旨在让用户快速开始训练自己的 Llama 模型。要使用自定义数据集，有两种可能的方式。第一种提供了一个在 .py 文件中返回数据集的函数，该函数可以提供给命令行工具。这不涉及更改 llama-recipes 的源代码。第二种方式是针对扩展 llama-recipes 的贡献，因为它涉及更改源代码。

### 在自定义数据上训练
要提供自定义数据集，您需要提供一个包含以下签名的单个 .py 文件的函数：
```@python
def get_custom_dataset(dataset_config, tokenizer, split: str):
```
有关 `get_custom_dataset` 的示例，可以查看 llama_recipes.datasets 中提供的数据集或 [examples/custom_dataset.py](../examples/custom_dataset.py)。上述签名中的 `dataset_config` 将是通过命令行进行修改的 llama_recipes.configs.dataset.custom_dataset 的实例。split 信号是否返回训练或验证数据集。默认函数名为 `get_custom_dataset`，但可以按照下面的描述更改。

为了启动带有自定义数据集的训练，我们需要设置 `--dataset` 以及 `--custom_dataset.file` 参数。
```
python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py" [TRAINING PARAMETERS]
```
要更改 .py 中使用的函数名称，可以追加冒号后面的名称，如下所示：
```
python

 -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py:get_foo" [TRAINING PARAMETERS]
```
这将在检索数据集时调用 `get_foo` 函数，而不是 `get_custom_dataset`。

### 添加新的数据集
每个数据集在 [configs/datasets.py](../src/llama_recipes/configs/datasets.py) 中都有一个相应的配置（数据类），其中包含数据集名称、训练/验证拆分名称以及可选的参数（例如数据文件等）。

此外，在 [datasets](../src/llama_recipes/datasets) 文件夹中，每个数据集都有一个预处理函数。数据集的返回数据需要能够通过调用 ```model(**data)``` 来在微调模型的前向方法中使用。对于 CausalLM 模型，这通常意味着数据需要以字典的形式存在，包含 "input_ids"、"attention_mask" 和 "labels" 字段。

要添加自定义数据集，需要执行以下步骤。

1. 根据上述模式创建数据集配置。示例可以在 [configs/datasets.py](../src/llama_recipes/configs/datasets.py) 中找到。
2. 创建一个预处理例程，该例程加载数据并返回 PyTorch 风格的数据集。预处理函数的签名需要是 (dataset_config, tokenizer, split_name)，其中 split_name 将是在数据类中定义的 train/validation 拆分的字符串。
3. 在 [utils/dataset_utils.py](../src/llama_recipes/utils/dataset_utils.py) 中的 DATASET_PREPROC 字典中将数据集名称和预处理函数注册为键和值。
4. 在训练配置中设置数据集字段为数据集名称，或使用 `llama_recipes.finetuning` 模块或 examples/finetuning.py 训练脚本的 --dataset 选项。

## 应用
下面我们列出了其他数据集及其主要用途，可用于微调。

### 问答，这些也可以用于评估
- [MMLU](https://huggingface.co/datasets/lukaemon/mmlu/viewer/astronomy/validation)
- [BoolQ](https://huggingface.co/datasets/boolq)
- [NarrativeQA](https://huggingface.co/datasets/narrativeqa)
- [NaturalQuestions](https://huggingface.co/datasets/natural_questions)（闭卷）
- [NaturalQuestions](https://huggingface.co/datasets/openbookqa)（开卷）
- [QuAC](https://huggingface.co/datasets/quac)
- [HellaSwag](https://huggingface.co/datasets/hellaswag)
- [OpenbookQA](https://huggingface.co/datasets/openbookqa)
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa)（可以帮助进行事实检查/模型的错误信息）

### 指令微调
- [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned)	5.2k	指令微调
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 1.5k	指令微调

### 用于快速测试的简单文本生成
[English](https://huggingface.co/datasets/Abirate/english_quotes) 引用	2508	多标签文本分类、文本生成

### 推理评估
- [bAbI](https://research.facebook.com/downloads/babi/)
- [Dyck](https://huggingface.co/datasets/dyk)
- [GSM8K](https://huggingface.co/datasets/gsm8k)
- [MATH](https://github.com/hendrycks/math)
- [APPS](https://huggingface.co/datasets/codeparrot/apps)
- [HumanEval](https://huggingface.co/datasets/openai_humaneval)
- [LSAT](https://huggingface.co/datasets/dmayhem93/agieval-lsat-ar)
- [实体匹配](https://huggingface.co/datasets/lighteval/EntityMatching)

### 毒性评估
- [Real_toxic_prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)

### 偏见评估
- [Crows_pair](https://huggingface.co/datasets/crows_pairs) 性别偏见
- WinoGender 性别偏见

### 有用的链接
有关评估数据集的更多信息，请查看[HELM](https://crfm.stanford.edu/helm/latest/)。