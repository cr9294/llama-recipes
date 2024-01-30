# 模型微调

## llama_recipes 微调脚本

提供了 `llama_recipes.finetuning` 模块或 `examples/finetuning.py` 脚本，用于模型的微调。微调脚本支持通过 `dataset` 参数选择三个数据集之一：`grammar_dataset`、`alpaca_dataset` 和 `samsum_dataset`。此外，还通过 [examples/custom_dataset.py](../examples/custom_dataset.py) 提供了 OpenAssistant/oasst1 数据集的示例，用作自定义数据集。请注意，使用任何数据集都应符合数据集的基础许可证（包括但不限于非商业用途）。

* [grammar_dataset](https://huggingface.co/datasets/jfleg) 包含 15 万对英语句子及其可能的更正。
* [alpaca_dataset](https://github.com/tatsu-lab/stanford_alpaca) 提供 5.2 万个由 `text-davinci-003` 生成的指令-响应对。
* [samsum_dataset](https://huggingface.co/datasets/samsum) 包含约 1.6 万个类似于 Messenger 的对话，带有摘要。
* [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1/) 包含来自助手式对话的约 8.8 万条消息。

## 批处理策略

Llama-recipes 支持两种将请求组合在一起的策略。默认设置为 `packing`，它将标记化的样本连接成长序列，填充模型的上下文长度。这是最高效的变体，因为它避免了任何填充，所有序列具有相同的长度。上下文长度边界上的样本被截断，截断序列的剩余部分被用作下一个长序列的起始。

如果训练数据量较小，这种过程可能会引入大量噪音到训练数据中，这可能会损害微调模型的预测性能。因此，我们还支持一种 `padding` 策略，它不引入由于截断序列而引入的额外噪音。该策略试图通过将相似长度的样本批处理在一起来最小化效率损失，因此只需最小的填充。

可以通过命令行参数 `--batching_strategy [packing]/[padding]` 选择批处理策略。

## 使用自定义数据集

llama-recipes中提供的数据集列表旨在让用户快速开始训练其Llama模型。要使用自定义数据集，有两种可能的方法。
第一种方法在 .py 文件中提供返回数据集的函数，该文件可以提供给命令行工具。
这不涉及更改 llama-recipes 的源代码。
第二种方法是针对扩展 llama-recipes 的贡献，因为它涉及更改源代码。

### 在自定义数据上进行训练

要提供自定义数据集，您需要提供一个包含以下签名的 .py 文件中的函数：

```@python
def get_custom_dataset(dataset_config, tokenizer, split: str):
```

您可以在 llama_recipes.datasets 或 [examples/custom_dataset.py](../examples/custom_dataset.py) 中找到 `get_custom_dataset` 的示例。
上述签名中的 `dataset_config` 将是通过命令行进行的修改的 `llama_recipes.configs.dataset.custom_dataset` 的实例。
`split` 信号表示是返回训练还是验证数据集。
默认函数名称为 `get_custom_dataset`，但可以根据以下说明更改。

为了开始使用自定义数据集进行训练，我们需要设置 `--dataset` 以及 `--custom_dataset.file` 参数。

```bash
python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py" [TRAINING PARAMETERS]
```

要更改在 .py 中使用的函数名称，可以附加名称，如下所示：

```bash
python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py:get_foo" [TRAINING PARAMETERS]
```

这将在检索数据集时调用 `get_foo` 函数，而不是 `get_custom_dataset`。

### 添加新数据集

每个数据集在 [configs/datasets.py](../src/llama_recipes/configs/datasets.py) 中都有一个相应的配置（dataclass），其中包含数据集名称、训练/验证拆分名称以及可选参数（如数据文件等）。

此外，每个数据集在 [datasets](../src/llama_recipes/datasets) 文件夹中都有一个预处理函数。
数据集的返回数据需要通过调用 ```model(**data)``` 的微调模型的前向方法可消耗。
对于 CausalLM 模型，这通常意味着数据需要以字典形式呈现，具有 "input_ids"、"attention_mask" 和 "labels" 字段。

要添加自定义数据集，需要执行以下步骤。

1. 根据上述描述创建数据集配置。示例可以在 [configs/datasets.py](../src/llama_recipes/configs/datasets.py) 中找到。
2. 创建一个预处理例程，该例程加载数据并返回一个 PyTorch 风格的数据集。预处理函数的签名需要是（dataset_config，tokenizer，split_name），其中 split_name 将是数据类中定义的 train/validation 拆分的字符串。
3. 将数据集名称和预处理函数注册到 [utils/dataset_utils.py](../src/llama_recipes/utils/dataset_utils.py) 中的 DATASET_PREPROC 字典中，通过将其插入为键和值。
4. 将训练配置中的 dataset 字段设置为数据集名称，或使用 `llama_recipes.finetuning` 模块或 examples/finetuning.py 训练脚本的 `--dataset` 选项。

## 应用

下面列出了可用于微调的其他数据集

示例。

### SuperGLUE

[SuperGLUE](https://huggingface.co/datasets/superglue) 是一组多样的自然语言理解任务的超级任务，包括自然语言推理、词义相似度和文本蕴含等任务。

```bash
# 下载数据集
python examples/download_superglue.py
# 运行微调
python -m llama_recipes.finetuning --dataset "super_glue" --super_glue.task "boolq" --super_glue_split "train" [TRAINING PARAMETERS]
```

请确保在运行前阅读配置文件以及用于特定任务的超级胶数据集的文档，以了解更多信息。

### IWSLT'14

[IWSLT'14 数据集](https://huggingface.co/datasets/iwslt14) 是一个用于机器翻译的小型数据集，包含英语和德语之间的句子对。

```bash
# 运行微调
python -m llama_recipes.finetuning --dataset "iwslt_14" --iwslt_14.src_lang "en" --iwslt_14.tgt_lang "de" --iwslt_14_split "train" [TRAINING PARAMETERS]
```

### SST-2

[SST-2](https://huggingface.co/datasets/sst) 数据集是一组包含电影评论的二分类情感分类任务。

```bash
# 运行微调
python -m llama_recipes.finetuning --dataset "sst_2" --sst_2_split "train" [TRAINING PARAMETERS]
```

### XNLI

[XNLI 数据集](https://huggingface.co/datasets/xnli) 是一个多语言自然语言推理数据集，包括 15 种不同语言的约 11 万对文本。

```bash
# 运行微调
python -m llama_recipes.finetuning --dataset "xnli" --xnli.lang "zh" --xnli_split "train" [TRAINING PARAMETERS]
```

请注意，上述命令中的 [TRAINING PARAMETERS] 应该替换为您选择的微调参数。有关支持的微调参数的更多信息，请参阅 finetuning.py 或 llama_recipes.configs 下的配置文件。