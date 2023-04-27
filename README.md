# MiniTorch 记录

原文：

<img src="https://minitorch.github.io/minitorch.svg" width="50%px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module0.html



## P0

- finish in 2023/04/13

### 实现

- operator：按照要求即可，主要问题在于精度问题
- module：简单的递归函数



因为难度较低，就不详细记录了，具体可以参考源码：

- `minitorch/operators.py`
- `minitorch/module.py`
- `minitorch/operators.py`



## P1

### Task1.1:中心差分

------

PS:上来就是高数，人就晕了

----

算法原理参考详情：[数值方法（十一）中心差分法数值微分求解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/361234743)

`minitorch.autodiff.central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-06) -> Any`

- f:$f(x) = y \ x\in \R^n, y\in \R$
- *vals:n个浮点数$x_0, ..., x_{n-1}$
- arg:偏导数自变量的索引i
- epsilon：无穷小量

核心思路：

- $x = vals[arg]$

- 求出$f(x+\epsilon)、 f(x - \epsilon)$
- 利用$\frac {\part f} {\part x} \approx \frac{f(x+\epsilon) - f(x - \epsilon) }{2 * \epsilon}$

通过了`test\_scalar.py`的测试



### Task 1.2 	
