# 常见问题解答（FAQ）

以下是我们经常遇到且认为在使用过程中可能有用的常见问题解答。

1. FSDP 是否支持一个 FSDP 单元中的混合精度？也就是说，在一个 FSDP 单元中，有些参数是 Fp16/Bf16，而另一些参数是 FP32。

    FSDP 要求每个 FSDP 单元具有一致的精度，因此目前不支持这种情况。这可能会在将来添加，但目前没有确定的时间表。

2. FSDP 如何处理混合的梯度要求？

    FSDP 不支持在一个 FSDP 单元中混合使用 `require_grad`。这意味着如果要冻结某些层，需要在 FSDP 单元级别而不是模型层级别上执行。例如，假设我们的模型有 30 个解码器层，我们想要冻结底部 28 层，并且只训练 2 个顶部的变压器层。在这种情况下，我们需要确保顶部两个变压器层的 `require_grad` 被设置为 `True`。

3. FSDP 如何与 PEFT 方法一起处理梯度要求/层冻结？

    我们在 auto_wrapping 策略中单独包装 PEFT 模块，这将导致 PEFT 模型具有 `require_grad=True`，而模型的其余部分为 `require_grad=False`。

4. 我可以添加自定义数据集吗？

    是的，您可以在[这里](Dataset.md)找到有关如何执行的更多信息。

5. 部署这些模型的硬件 SKU 要求是什么？

    硬件要求因延迟、吞吐量和成本限制而异。为了获得良好的延迟，模型使用了 NVIDIA A100 或 H100 机器上的张量并行ism 分割成多个 GPU。但也可以使用 TPUs、其他类型的 GPU，如 A10G、T4、L4，甚至是商用硬件（例如 https://github.com/ggerganov/llama.cpp）。如果在 CPU 上工作，可以查看英特尔的[博客文章](https://www.intel.com/content/www/us/en/developer/articles/news/llama2.html)，了解 Llama 2 在 CPU 上的性能。

6. 微调 Llama 预训练模型的硬件 SKU 要求是什么？

    微调要求根据数据量、完成微调的时间和成本限制而异。为了微调这些模型，我们通常使用多个 NVIDIA A100 机器，跨节点进行数据并行ism，以及节点内的数据和张量并行ism 的混合。但使用单个机器，或其他类型的 GPU，如 NVIDIA A10G 或 H100，也是完全可能的（例如 alpaca 模型是在单个 RTX4090 上训练的：https://github.com/tloen/alpaca-lora）。

7. 在微调期间如何处理可能导致 OOM 的 CUDA 内存碎片化？

    在某些情况下，您可能会发现在模型检查点之后（特别是使用 FSDP 时，这通常不会发生在 PEFT 方法中），CUDA 内存的已保留和已分配部分已增加。这可能是由于 CUDA 内存碎片化引起的。PyTorch 最近添加了一个环境变量，有助于更好地管理内存碎片化（截至文档撰写时，即 2023 年 7 月 30 日，此功能在 PyTorch nightly 中可用）。您可以在主要训练脚本中设置这个变量，如下所示：

    ```bash
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    ```
    我们还在 [train_utils.py](../src/llama_recipes/utils/train_utils.py) 的 `setup_environ_flags` 中添加了这个环境变量，请根据需要取消注释。

8. 是否有额外的调试标志？

    环境变量 `TORCH_DISTRIBUTED_DEBUG` 可以用于触发额外的有用日志记录和集体同步检查，以确保所有秩适当同步。`TORCH_DISTRIBUTED_DEBUG` 可以设置为 OFF（默认值）、INFO 或 DETAIL，具体取决于所需的调试级别。请注意，最详细的选项 DETAIL 可能会影响应用程序的性能，因此只有在调试问题时才应使用。

    我们还在 [train_utils.py](../src/llama_recipes/utils/train_utils.py) 的 `setup_environ_flags` 中添加了这个环境变量，请根据需要取消注释。

9. 运行推理时我遇到了导入错误。

    验证您的 CUDA 环境变量在您的机器上是否正确设置。例如，对于 bitsandbytes，您通常可以将其设置如下以使其在 AWS 上的 A100 80g 上正常工作。

    ```bash
    export CUDA_HOME="/usr/local/cuda-11.8"
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/efa/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
    ```