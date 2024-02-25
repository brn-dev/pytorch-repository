### Non-linear Activations in PyTorch (2.2)

1. **nn.ELU**: Applies the Exponential Linear Unit (ELU) function, element-wise, as described in the paper: "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".

2. **nn.Hardshrink**: Applies the Hard Shrinkage (Hardshrink) function element-wise.

3. **nn.Hardsigmoid**: Applies the Hardsigmoid function element-wise.

4. **nn.Hardtanh**: Applies the HardTanh function element-wise.

5. **nn.Hardswish**: Applies the Hardswish function, element-wise, as described in the paper: "Searching for MobileNetV3".

6. **nn.LeakyReLU**: Applies the element-wise function: Leaky ReLU.

7. **nn.LogSigmoid**: Applies the element-wise function: LogSigmoid.

8. **nn.MultiheadAttention**: Allows the model to jointly attend to information from different representation subspaces as described in the paper: "Attention Is All You Need".

9. **nn.PReLU**: Applies the element-wise function: Parametric ReLU.

10. **nn.ReLU**: Applies the rectified linear unit function element-wise.

11. **nn.ReLU6**: Applies the element-wise function: ReLU6.

12. **nn.RReLU**: Applies the randomized leaky rectified linear unit function, element-wise.

13. **nn.SELU**: Applied element-wise as: Scaled Exponential Linear Unit (SELU).

14. **nn.CELU**: Applies the element-wise function: Continuous Exponential Linear Unit (CELU).

15. **nn.GELU**: Applies the Gaussian Error Linear Units (GELU) function.

16. **nn.Sigmoid**: Applies the element-wise function: Sigmoid.

17. **nn.SiLU**: Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

18. **nn.Mish**: Applies the Mish function, element-wise.

19. **nn.Softplus**: Applies the Softplus function: `Softplus(x) = (1/β) * log(1 + exp(β * x))` element-wise.

20. **nn.Softshrink**: Applies the soft shrinkage function elementwise.

21. **nn.Softsign**: Applies the element-wise function: Softsign.

22. **nn.Tanh**: Applies the Hyperbolic Tangent (Tanh) function element-wise.

23. **nn.Tanhshrink**: Applies the element-wise function: Tanhshrink.

24. **nn.Threshold**: Thresholds each element of the input Tensor.

25. **nn.GLU**: Applies the gated linear unit function: `GLU(a, b) = a ⊗ σ(b)` where `a` is the first half of the input matrices and `b` is the second half.
