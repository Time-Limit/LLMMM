# 开发计划
本项目是一个针对 0.5~7B LLM 的推理框架，旨在通过 kernel fusion 提升推理速度。目前还在开发中。

因为作者只有一块 4090 用来开发，所以目前只能保证本项目的代码能在 SM89 上跑通。

作者是一名 CUDA 的初学者，因此本项目的开发计划可能会有调整，但整体上会有以下几部分。

* Matrix Multiplication
  * FP32
  * FP16
  * FP8 Blockwise Quant
* Attention
  * MHA/GQA
  * FP16/FP8
  * Tree Mask
* 尝试将 RMSNorm、fp8 dynamic quant、SILU、ROPE等算子融合进 MM 或者 Attention
* 补充 TopP/TopK，Embedding Lookup 等流程，跑通 Qwen3 4B 模型。
* 支持投机采样
* 支持 OpenAI

# 进度
目前还在 MM 部分的收尾阶段，还需要在 FP16/FP8 版本上支持 SPLIT K、Auto Tune。
## FP32 
虽然 FP32 的 MM 不会用到最终的推理框架上，但作为入门第一课还是值得一写的。

当前版本的性能基本和 cublas 持平了。优化过程记录在了 https://zhuanlan.zhihu.com/p/1912906578081842292。

<img width="1134" height="644" alt="image" src="https://github.com/user-attachments/assets/915c7b43-92d5-4448-86f5-5c9065afac7e" />
<img width="1120" height="651" alt="image" src="https://github.com/user-attachments/assets/33cb36d4-8908-47ba-94b6-03157429f16f" />

## FP16
因尚未支持 SPLIT K，所以先只比较了 M=N=K=4096 的性能，以大约 0.5us 的优势略快于 cublas。
<img width="803" height="329" alt="image" src="https://github.com/user-attachments/assets/0573cc78-1c31-42d9-b536-81382ef8a6b2" />
<img width="923" height="260" alt="image" src="https://github.com/user-attachments/assets/9406021c-43c4-4b08-b11d-f4c6684be537" />

算力利用率也来到了 97.7% (按 Peak=146.358TFLOPS 计算)，应该算是比较高的利用率了吧。
<img width="1484" height="704" alt="image" src="https://github.com/user-attachments/assets/384283de-5723-4302-b7b5-aa2bed9df518" />

## FP8
融合了 C 矩阵的 blockwise quant。在 M=N=K=4096 上，算力利用率大约有 79.3% 的利用率(按 Peak=293TFLOPS 计算）。
<img width="1452" height="381" alt="image" src="https://github.com/user-attachments/assets/dab3dd99-8365-4088-8efa-ae015ac53a4d" />

以在 4B 模型为例，融合 C 矩阵的 blockwise quant，大概可以加速 5%。
