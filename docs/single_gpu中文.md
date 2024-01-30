# 使用单个 GPU 进行微调

为了在单个 GPU 上运行微调，我们将使用两个包

1- [PEFT](https://huggingface.co/blog/peft) 方法，具体使用 HuggingFace 的 [PEFT](https://github.com/huggingface/peft) 库。

2- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) int8 量化。

在 PEFT 和 Int8 量化的组合下，我们将能够在单个消费级 GPU（如 A10）上微调 Llama 2 7B 模型。

## 要求
要运行示例，请确保安装了 llama-recipes 包（有关详细信息，请参见 [README.md](../README.md)）。

**请注意，llama-recipes 包将安装 PyTorch 2.0.1 版本，如果要运行 FSDP + PEFT，请确保安装 PyTorch nightlies。**

## 如何运行？

获取一台带有一个 GPU 的机器的访问权限，或者如果使用多 GPU 机器，请确保只使其中一个 GPU 可见，使用 `export CUDA_VISIBLE_DEVICES=GPU:id` 命令运行以下命令。默认情况下，它使用 `samsum_dataset` 进行摘要应用。

```bash
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```

上面命令中使用的参数是：

* `--use_peft` 布尔标志，用于在脚本中启用 PEFT 方法。

* `--peft_method` 用于指定 PEFT 方法，这里我们使用 `lora`，其他选项包括 `llama_adapter`、`prefix`。

* `--quantization` 布尔标志，用于启用 int8 量化。

## 如何使用不同的数据集运行？

当前支持 4 个数据集，可以在 [数据集配置文件](../src/llama_recipes/configs/datasets.py) 中找到。

* `grammar_dataset`：使用[此笔记本](../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb)拉取和处理 Jfleg 和 C4 200M 数据集，用于语法检查。

* `alpaca_dataset`：要获取此开源数据，请下载 `aplaca.json` 到 `ft_dataset` 文件夹。

```bash
wget -P src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

要使用每个数据集运行，请在命令中设置 `dataset` 标志，如下所示：

```bash
# grammer_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

## 在哪里配置设置？

* [训练配置文件](../src/llama_recipes/configs/training.py)是主要的配置文件，用于指定我们运行的设置，可以在

它让我们指定训练设置，从 `model_name` 到 `dataset_name`、`batch_size` 等都可以在这里设置。以下是支持的设置列表：

```python
model_name: str="PATH/to/LLAMA 2/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
gradient_accumulation_steps: int=1
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset,grammar_dataset
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False
```

* [数据集配置文件](../src/llama_recipes/configs/datasets.py)提供了数据集的可用选项。

* [peft 配置文件](../src/llama_recipes/configs/peft.py)提供了支持的 PEFT 方法和相应的设置，可以进行修改。