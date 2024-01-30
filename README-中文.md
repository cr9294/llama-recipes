# Llama 2 微调 / 推理配方、示例、基准测试和演示应用

**[更新于2023年12月28日] 我们为我们的示例推理脚本添加了对 Llama Guard 的支持，作为安全检查器，并且还通过一个示例脚本和提示格式提供了独立推理的支持。更多详细信息请查看[这里](./examples/llama_guard/README.md)。关于为 Llama Guard 进行微调数据格式的详细信息，我们提供了一个脚本和示例用法，可在[这里](./src/llama_recipes/data/llama_guard/README.md)找到。**

**[更新于2023年12月14日] 我们最近发布了一系列 Llama 2 演示应用，可在[这里](./demo_apps)找到。这些应用展示了如何在本地、云端或本地运行 Llama（locally, in the cloud, or on-prem），如何使用 Azure Llama 2 API（模型即服务），如何向 Llama 提问（一般性或关于自定义数据，如 PDF、DB 或实时数据），如何将 Llama 集成到 WhatsApp 和 Messenger 中，以及如何使用 RAG（检索增强生成）实现端到端聊天机器人。**

'llama-recipes' 仓库是[Llama 2 模型](https://github.com/facebookresearch/llama)的伴侣。此仓库的目标是提供示例，以便快速开始进行领域自适应的微调，并演示如何运行微调模型的推理。为了方便使用，示例使用 Hugging Face 转换版本的模型。有关模型转换的步骤，请参见[此处](#model-conversion-to-hugging-face)。

此外，我们还提供了许多演示应用，展示了 Llama 2 的使用以及其他生态系统解决方案，以在本地、云端和本地运行 Llama 2。

Llama 2 是一项新技术，使用时存在潜在风险。迄今为止的测试未能涵盖所有场景。为了帮助开发人员应对这些风险，我们创建了[负责任使用指南](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf)。更多详细信息也可以在我们的研究论文中找到。要下载模型，请按照[Llama 2 仓库](https://github.com/facebookresearch/llama)上的说明进行操作。

# 目录
1. [快速入门](#quick-start)
2. [模型转换](#model-conversion-to-hugging-face)
3. [微调](#fine-tuning)
    - [单 GPU](#single-gpu)
    - [多 GPU 单节点](#multiple-gpus-one-node)
    - [多 GPU 多节点](#multi-gpu-multi-node)
4. [推理](./docs/inference.md)
5. [演示应用](#demo-apps)
6. [仓库组织](#repository-organization)
7. [许可和可接受使用政策](#license)

# 快速入门

[Llama 2 Jupyter 笔记本](./examples/quickstart.ipynb): 此 Jupyter 笔记本介绍了如何使用 [samsum](https://huggingface.co/datasets/samsum) 对 Llama 2 模型进行文本摘要任务的微调。该笔记本使用参数高效微调（PEFT）和 int8 量化，在像 A10 这样的单 GPU（24GB GPU 内存）上微调了 7B 模型。

# 安装
Llama-recipes 提供了一个 pip 分发包，便于在其他项目中安装和使用。或者，也可以从源代码安装。

## 使用 pip 安装
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```

## 安装可选依赖项
Llama-recipes 提供了可选软件包的安装。有三个可选的依赖组。
要运行单元测试，可以使用以下命令安装所需的依赖项：
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[tests]
```
对于 vLLM 示例，我们需要额外的要求，可以使用以下命令安装：
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[vllm]
```
要使用敏感主题安全检查器，请使用以下命令安装：
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[auditnlg]
```
可选依赖项也可以与 [option1,option2] 组合使用。

## 从源代码安装
要从源代码安装，例如进行开发，请使用以下命令。我们使用 hatchling 作为我们的构建后端，它需要一个最新的 pip 以及 setuptools 包。
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```
对于开发和为 llama-recipes 贡献，请安装所有可选依赖项：
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .[tests,auditnlg,vllm]
```

⚠️ **注意** ⚠️  某些功能（特别是使用 FSDP + PEFT 进行微调）目前需要安装 PyTorch nightlies。如果使用这些功能，请确保按照[此指南](https://pytorch.org/get-started/locally/)安

装 nightlies。

**注意** 所有在[配置文件](src/llama_recipes/configs/)中定义的设置在运行脚本时可以作为参数通过 CLI 传递，无需直接从配置文件更改。

**有关更详细的信息，请查看以下内容：**

* [单 GPU 微调](./docs/single_gpu.md)
* [多 GPU 微调](./docs/multi_gpu.md)
* [LLM 微调](./docs/LLM_finetuning.md)
* [添加自定义数据集](./docs/Dataset.md)
* [推理](./docs/inference.md)
* [评估工具](./eval/README.md)
* [常见问题](./docs/FAQ.md)

# 模型在哪里找到？

您可以在 Hugging Face hub 上找到 Llama 2 模型，链接在[这里](https://huggingface.co/meta-llama)，带有名称中的 `hf` 的模型已经转换为 Hugging Face checkpoints，因此无需进一步转换。下面的转换步骤仅适用于在 Hugging Face 模型 hub 上托管的来自 Meta 的原始模型权重。

# 将模型转换为 Hugging Face
此文件夹中的配方和笔记本使用了 Hugging Face 的 transformers 库提供的 Llama 2 模型定义。

鉴于原始检查点位于 models/7B 下，您可以安装所有要求并执行以下命令将检查点转换为 Hugging Face 格式：

```bash
## 从源代码安装 Hugging Face Transformers
pip freeze | grep transformers ## 验证版本是否为4.31.0或更高

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

# 微调

为了微调 Llama 2 模型以适应您的领域特定用例，配方中包含了 PEFT、FSDP、PEFT+FSDP 的示例以及一些测试数据集。详情请参见 [LLM 微调](./docs/LLM_finetuning.md)。

## 单 GPU 和多 GPU 微调

如果您想直接进行单 GPU 或多 GPU 微调，请在像 A10、T4、V100、A100 等的单 GPU 上运行以下示例。以下示例和配方中的所有参数需要根据手头的模型、方法、数据和任务进行进一步调整，以获得所需的结果。

**注意：**
* 在以下命令中更改数据集，请传递 `dataset` 参数。集成数据集的当前选项有 `grammar_dataset`、`alpaca_dataset` 和 `samsum_dataset`。此外，我们将 OpenAssistant/oasst1 数据集集成为[自定义数据集的示例](./examples/custom_dataset.py)。关于如何使用自己的数据集以及如何添加自定义数据集的说明可以在 [Dataset.md](./docs/Dataset.md#using-custom-datasets) 中找到。对于 `grammar_dataset`、`alpaca_dataset` 请确保使用[此处](./docs/single_gpu.md#how-to-run-with-different-datasets)的建议说明进行设置。

* 默认数据集和其他 LORA 配置已设置为 `samsum_dataset`。

* 确保在[训练配置](src/llama_recipes/configs/training.py)中设置正确的模型路径。

* 要保存用于评估的损失和困惑度度量，请通过将 `--save_metrics` 传递给微调脚本来启用此功能。可以使用 [plot_metrics.py](./examples/plot_metrics.py) 脚本绘制文件，`python examples/plot_metrics.py --file_path path/to/metrics.json`

### 单 GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name /path_of_model_folder/7B --output_dir path/to/save/PEFT/model

```

这里我们使用参数高效方法（PEFT），如下一节所述。要运行上述命令，请确保传递 `peft_method` 参数，可以设置为 `lora`、`llama_adapter` 或 `prefix`。

**注意** 如果在带有多个 GPU 的机器上运行，请确保只使用 `export CUDA_VISIBLE_DEVICES=GPU:id` 使其中一个 GPU 可见。

**确保设置 `save_model` 参数以保存模型。请确保检查[训练配置](src/llama_recipes/configs/training.py)中的其他训练参数以及需要的配置文件夹中的其他参数。所有参数都可以作为参数传递给训练脚本，无需更改配置文件。**

### 多 GPU 单节点:

**注意** 请确保使用 PyTorch Nightlies 来使用 PEFT+FSDP。另外，目前 FSDP 不支持来自 bit&bytes 的 int8 量化。

```bash

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /path_of_model_folder/7B --fsdp_config.pure_bf16 --output_dir path/to/save/PEFT/model

```

这里我们使用了 FSDP，如下一节所述，可以与 PEFT 方法一起使用。为了使用 FSDP 运行 PEFT 方法，请确保传递 `use_peft` 和 `peft_method` 参数以及 `enable_fsdp`。这里我们使用 `BF16` 进行训练。

## Flash Attention 和 Xformer 内存高效内核

设置 `use_fast_kernels` 将启用 Flash Attention 或基于所使用硬件的 Xformer 内存高效内核。这将加速微调作业。这已在 Hugging Face 的 `optimum` 库中作为一个简单的 API 启用，请在[此处](https://pytorch.org/blog/out-of-the-box-acceleration/)了解更多信息。

```bash
torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /path_of_model_folder/7B --fsdp_config.pure_bf16 --output_dir path/to/save/PEFT/model --use_fast_kernels
```

### 仅使用 FSDP 进行微调

如果您想运行完整参数微调而不使用 PEFT 方法，请使用以下命令。确保将 `nproc_per_node` 更改为您可用的 GPU 数量。此选项已在 8xA100，40GB GPU 上使用 `BF16` 进行测试。

```bash

torchrun --nnodes 1 --nproc_per_node 

8  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --use_fast_kernels

```

### 使用 FSDP 在 70B 模型上进行微调

如果您希望在 70B 模型上运行完整参数微调，可以在以下命令中启用 `low_cpu_fsdp` 模式。此选项将仅在将模型移动到设备之前将模型加载到 rank0 上，以构造 FSDP。这在加载大型模型（如 70B 模型）时可以显著节省 CPU 内存（在 8-GPU 节点上，这将将 70B 模型的 CPU 内存从 2+T 减少到 280G）。此选项已在 16xA100，80GB GPU 上使用 `BF16` 进行测试。

```bash

torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --model_name /path_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

### 多 GPU 多节点:

```bash

sbatch multi_node.slurm
# 在运行之前，请在脚本中更改节点数和每个节点的 GPU 数量。

```
有关我们的微调策略的更多信息，请阅读[此处](./docs/LLM_finetuning.md)。

# 评估工具

在这里，我们使用来自 `EleutherAI` 的 `lm-evaluation-harness` 来评估微调的 Llama 2 模型。这也可以扩展到评估 Llama 2 模型推理的其他优化，比如量化。请使用[这个文档](./eval/README.md)开始。

# 演示应用程序
此文件夹包含一系列由 Llama2 驱动的应用程序：
* 快速启动 Llama 部署和与 Llama 的基本交互
1. 在您的 Mac 上使用 Llama 并询问 Llama 一般性问题
2. 在 Google Colab 上使用 Llama
3. 在云中使用 Llama，并向 Llama 提问有关 PDF 中的非结构化数据的问题
4. 在本地使用 vLLM 和 TGI 的 Llama
5. 使用 RAG（检索增强生成）的 Llama 聊天机器人
6. Azure Llama 2 API（模型即服务）

* 专业化的 Llama 用例：
1. 让 Llama 总结视频内容
2. 向 Llama 提问有关 DB 中的结构化数据的问题
3. 向 Llama 提问有关网络上的实时数据的问题
4. 构建支持 Llama 的 WhatsApp 聊天机器人

# 基准测试
此文件夹包含一系列用于在各种后端上对 Llama 2 模型推理进行基准测试的脚本：
1. 本地 - 流行的服务框架和容器（例如 vLLM）
2. （进行中）云 API - 流行的 API 服务（例如 Azure 模型即服务）
3. （进行中）在设备上 - 流行的 Android 和 iOS 上的本地推理解决方案（例如 mlc-llm、QNN）
4. （进行中）优化 - 用于更快推理和量化的流行优化解决方案（例如 AutoAWQ）

# 仓库组织
该存储库的组织方式如下：

[benchmarks](./benchmarks): 包含一系列用于在各种后端上对 Llama 2 模型推理进行基准测试的脚本。

[configs](src/llama_recipes/configs/): 包含 PEFT 方法、FSDP、数据集的配置文件。

[docs](docs/): 单 GPU 和多 GPU 微调的示例配方。

[datasets](src/llama_recipes/datasets/): 包含用于下载和处理每个数据集的单独脚本。注意：对数据集的使用应符合数据集的基础许可证（包括但不限于非商业用途）

[demo_apps](./demo_apps): 包含一系列由 Llama2 驱动的应用程序，从快速启动部署到如何询问 Llama 有关非结构化数据、结构化数据、实时数据和视频摘要的问题。

[examples](./examples/): 包含有关 Llama 2 模型微调和推理的示例脚本，以及如何安全使用它们的信息。

[inference](src/llama_recipes/inference/): 包括用于微调模型推理的模块。

[model_checkpointing](src/llama_recipes/model_checkpointing/): 包含 FSDP 检查点处理程序。

[policies](src/llama_recipes/policies/): 包含 FSDP 脚本，提供不同的策略，例如混合精度、变压器封装策略和激活检查点，以及任何精度优化器（用于以纯 bf16 模式运行 FSDP）。

[utils](src/llama_recipes/utils/): 用于：

- `train_utils.py` 提供训练/评估循环和更多训练实用程序。

- `dataset_utils.py` 用于获取预处理的数据集。

- `config_utils.py` 用于通过 CLI 覆盖从 CLI 接收的配置。

- `fsdp_utils.py` 提供用于 PEFT 方法的 FSDP 封装策略。

- `memory_utils.py` 上下文管理器，用于在训练循环中跟踪不同的内存统计信息。

# 许可证
请查看[此处](LICENSE)的许可证文件和[此处](USE_POLICY.md)的可接受使用政策。