###### 1. why positional encoding?

transformer的encoder和decoder的输入层中，使用了Positional Encoding，使得最终的输入满足：

```bash
input = input_embedding + positional_encoding
```

https://www.zhihu.com/people/lemonround/posts?page=3

kv缓存

https://zhuanlan.zhihu.com/p/662498827



softmax

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/





在大语言模型（LLM）中，`context_length`（上下文长度）指的是模型在处理输入时能够记住或“理解”的最大词或标记（tokens）数量。如果输入的文本长度超过这个限制，超出部分将会被截断，模型只会根据前面的一部分文本进行推理和生成。

不同的模型具有不同的`context_length`，例如：

-   GPT-3的`context_length`是2048个标记。
-   GPT-4在某些版本中可以支持更长的上下文，甚至可达8000个标记或更多。

`context_length`与模型的推理能力密切相关，较长的上下文允许模型在生成时考虑更多的历史信息，但也会带来更高的计算成本和内存需求。

总结来说，`context_length`决定了模型在每次生成中能够考虑多长的输入文本，从而影响其回答的连贯性和准确性。

###### 2. prefill和decode

https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5
