## quantization

#### 1. **SmoothQuant**

##### 基本概念

​	SmoothQuant是一种量化感知训练 (QAT) 和后训练量化 (PTQ) 方法的结合，用于减轻激活值和权重量化中的不匹配问题。
大语言模型中的激活值通常具有高动态范围，直接量化会导致严重的精度损失。SmoothQuant提出了平滑激活值的方法，通过重新分配权重和激活值的缩放因子来缓解这个问题。

##### 关键思想

1.  **激活-权重平衡 (Activation-Weight Equalization):**
    -   将激活值的动态范围部分转移到权重中，从而降低激活值的动态范围。
    -   实现方式是对权重矩阵按列进行缩放，对激活值按行进行反向缩放。
2.  **后训练量化 (PTQ):**
    -   SmoothQuant是无监督的后训练量化方法，可以在不需要额外数据标注的情况下对模型进行量化。

##### 优点

-   **适用于大模型：** SmoothQuant特别针对大语言模型（如 GPT、BERT）在硬件上的低比特量化加速。
-   **精度高：** 在INT8量化情况下能保留模型的大部分性能。
-   **简单：** 无需重新训练或标注数据。

#### **工作流程**

1.  通过统计权重和激活的动态范围，计算缩放因子。
2.  对激活值和权重进行动态范围重新分配。
3.  应用量化操作。

#### 2. **AWQ (Activation-Aware Weight Quantization)**

#### **基本概念**

​	AWQ是一种基于权重量化的方法，核心在于利用激活值的分布信息优化权重量化策略。它特别关注在大模型的低比特 (如 INT4) 表示下保持较高的推理性能。

##### 关键思想

1.  **激活感知 (Activation-Aware):**
    -   通过分析激活值在模型推理中的分布，设计量化策略以降低激活值对权重量化的敏感性。
    -   使用校准数据来捕捉激活值分布特性。
2.  **量化权重 (Weight Quantization):**
    -   AWQ 在权重量化时根据激活值的分布调整量化误差。
    -   对每个权重块单独计算量化参数（如零点和比例因子）。
3.  **分块量化 (Block-wise Quantization):**
    -   将大模型的权重矩阵划分为多个块，每块单独进行量化，从而提高硬件效率。

##### 优点

-   **低比特量化：** AWQ在INT4下也能保持较高的性能。
-   **模块化设计：** 可灵活应用于不同模型和架构。
-   **激活感知优化：** 对权重量化引入的误差更加鲁棒。

#### 3. SmoothQuant和AWQ的对比

| 特性                 | **SmoothQuant**                     | **AWQ**                                    |
| -------------------- | ----------------------------------- | ------------------------------------------ |
| **核心思想**         | 激活-权重平衡，通过缩放平滑激活值   | 激活感知权重量化，基于激活分布优化权重量化 |
| **适用范围**         | 主要用于INT8量化                    | 可扩展到INT4等更低比特                     |
| **是否需要校准数据** | 不需要                              | 需要少量校准数据                           |
| **复杂性**           | 较低，适用于快速量化                | 较高，需更多预处理                         |
| **精度表现**         | 在大语言模型中表现良好，尤其是 INT8 | 在低比特表示下（如INT4）更优               |
| **硬件友好性**       | 兼容常见硬件的INT8加速              | 针对更高效的硬件优化设计                   |

#### 4. **应用场景**

-   **SmoothQuant：**
    -   适合需要快速部署的场景，如在通用硬件（CPU、GPU）上对大模型进行推理加速。
    -   特别适用于数据不可用或隐私敏感的环境。
-   **AWQ：**
    -   适用于对模型精度要求高且硬件支持INT4的场景。
    -   需要部分校准数据，适合特定应用如边缘设备推理。

#### 总结

-   **SmoothQuant**更适合快速、简单的模型量化，尤其在INT8量化和无校准数据情况下表现优异。
-   **AWQ**侧重激活感知，能在更低比特（如 INT4）上实现出色的推理性能，适合更先进的硬件环境。

具体选择应根据模型、硬件和场景需求来决定。



#### 1. Quantization

##### 1.1 Zero-point Quantization

$$
\begin{align*}
\text{scale} &= \frac{255}{\max(X) - \min(X)} \\
\text{zeropoint} &= -\text{round}(\text{scale} \cdot \min(X)) - 128
\end{align*}
$$


$$
\begin{align*}
X_{\text{quant}} &= \text{round}(\text{scale} \cdot X + \text{zeropoint}) \\
X_{\text{dequant}} &= \frac{X_{\text{quant}} - \text{zeropoint}}{\text{scale}}
\end{align*}
$$


##### 1.2 Weight vs Activation Quantization

-   Weight quantization
    -   Store weights in INT8, dequantize into FP32 when running it
    -   Not faster inference, but saves space
-   Activation quantization
    -   Convert all inputs and outputs into INT8 and do computations in INT8
    -   Need calibration (static or dynamic) to determine scale factors for data at each layer





Greedy Search

Beam Search





https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6







Cources

-   [EfficientML.ai](https://efficientml.ai)