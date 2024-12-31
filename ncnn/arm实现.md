## arm实现

#### 1. Abs

Summary

Absolute takes one input data (Tensor) and produces one output data (Tensor) where absolute value, y = abs(x), is applied to the tensor elementwise.

`vld1q_f32`（Vector Load 1 Quadword of 32-bit Floating-Point）从内存地址 `ptr` 处开始加载 4 个连续的 32 位浮点数，并将它们存储在一个 128 位的寄存器中。

```cpp
float32x4_t vld1q_f32(const float *ptr);
```

绝对值实现

```cpp
int AbsVal_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vabsq_f32(_p);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "vld1.f32   {d0-d1}, [%1]!      \n"
            "vabs.f32   q0, q0              \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d1}, [%2]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(ptr),    // %1
              "=r"(outptr)  // %2
            : "0"(nn),
              "1"(ptr),
              "2"(outptr)
            : "cc", "memory", "q0"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *outptr = *ptr > 0 ? *ptr : -*ptr;

            ptr++;
            outptr++;
        }
    }

    return 0;
}
```



```bash
asm volatile(
    "0:                             \n"    // 标签 0: 用于标记循环的起始点
    "vld1.f32   {d0-d1}, [%1]!      \n"    // 从 ptr 指向的内存位置加载 4 个 32 位浮点数到 d0 和 d1（q0 中的低 64 位和高 64 位）
    "vabs.f32   q0, q0              \n"    // 计算 q0 中每个元素的绝对值
    "subs       %0, #1              \n"    // 将 nn 减 1，并更新条件标志
    "vst1.f32   {d0-d1}, [%2]!      \n"    // 将处理后的数据存储回 outptr
    "bne        0b                  \n"    // 如果 nn 不为 0，则跳转回标签 0，继续循环
    : "=r"(nn),     // %0: 输出 nn 的新值
      "=r"(ptr),    // %1: 输出 ptr 的新值（自增后的 ptr）
      "=r"(outptr)  // %2: 输出 outptr 的新值（自增后的 outptr）
    : "0"(nn),
      "1"(ptr),
      "2"(outptr)
    : "cc", "memory", "q0"  // 影响的寄存器：条件代码寄存器、内存、q0
);
```

https://developer.arm.com/documentation/102474/0100/Fundamentals-of-Armv8-Neon-technology/Registers--vectors--lanes-and-elements?lang=en