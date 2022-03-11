# Homework 1

## 实验目的

**能够使用CUDA编写基础算子，再通过pytorch进行调用。**


## 实验内容

实验依据课程进度，将被拆解成三个部分：

1. Part1——搭建pytorch+CUDA环境，学习用pytorch搭建神经网络
2. Part2——学习c++代码到python端的binding（Python调用C++代码）
3. Part3——学习基础的cuda编程（编写算子的cuda实现，并在pytorch端调用）


## Part1 : Pytorch Basics

**Just learn pytorch**

## Part 2-1: Customed Layer 自定义算子

> 详细内容参见官网教程：https://pytorch.org/docs/stable/notes/extending.html#extending-autograd

**自己实现算子的正向、反向传播**，即利用`torch.autograd.Function`和`torch.nn.Module` 自定义算子的`forward()`, `backward()`。自定义算子的步骤如下：

**Steps:**

1. 继承`torch.autograd.Funtion`类，并定义`forward()`, `backward()`方法。*他们都是`@staticmethod`*。

* `forward()`的参数可以是任何值。保存梯度的参数需要在传入前转换为不保存梯度的参数。可以返回一个`Tensor`或多个张量组成的`Tuple`。

* `backward()`定义了求梯度的规则。`backward()`的输出参数个数与`forward()`的输入参数个数相同，并且`backward()`的输出是其对应的`forward()`输入参数的梯度。如下例子，`forward()`输入`input`和`weight`，`backward()`输出则应有`grad_input`和`grad_weight`。**`backward()`中不可以使用In-place操作**

```python
class myLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight) # 保存input和weight，在backward()回使用
        output = input.mm(weight.t())
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors # 读取ctx保存的Tensor
        grad_input = grad_weight = None
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight
```

2. 合理使用`ctx`所带的函数

* `ctx.save_for_backward()`必须在保存`forward()`的输入或输出张量，以便在`backward()`中使用。

* `ctx.mark_dirty()` 用来标记出在`forward()`中被In-place修改的变量。

* `ctx.mark_non_differentiable()` 用标明输出时不可微分的。默认情况下所有的输出都是可微分的。

* `ctx.set_materialize_grads()`

3. 如果自定义的`Funtion`不支持 _double backward_，则使用`once_differentiable()`来修饰backward()。

4. 使用`torch.autograd.gradcheck()`检验自定义算子是否正确。

## Part 2-2: CPP Binding

> CPP Binding官网教程：https://pybind11.readthedocs.io/en/stable/

### 打包C++的脚本文件：`setup.py`
本部分实现了**1.使用C++定义函数，2.打包成Python module 3.在Python中调用C++函数**。

[`setup.py`](https://github.com/guanrenyang/AI3615-AI-Chip-Design/blob/main/hw1/2.%20customed%20layer%20%2B%20cpp%20binding/2.%20cpp%20binding/setup.py)的作用是 _安装_ `.cpp`文件，将一个`.cpp`文件打包为Python Module。

### cpp文件内的声明：`PYBIND11_MODULE`
[`mylinear.cpp`](https://github.com/guanrenyang/AI3615-AI-Chip-Design/blob/main/hw1/2.%20customed%20layer%20%2B%20cpp%20binding/2.%20cpp%20binding/mylinear.cpp)中的`PYBIND11_MODULE`语句将一个`cpp`函数 _注册_ 为Python函数。

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylinear_forward, "myLinear forward"); 
  // forward为Python函数名
  // mylinear_forward为CPP函数名
  // "myLinear forward"为对函数的描述(仅推测)
  m.def("backward", &mylinear_backward, "myLinear backward");
  m.def("cppprint", &my_cppprint, "my cpp print");
}
```

本次作业实现在`mylinear.cpp`文件中实现`my_cppprint`函数，在`mylinear.py`的`backward()`中调用。

### 两种打包方式：`install`和`develop`

* `install` 将C++打包成库并安装进Python中。*类似于一般`pip`安装库时的方式*。适合于不需要对C++库进行修改时使用。
* `develop` 将C++打包成本地的动态链接库，适合经常要对C++文件进行修改时使用。

## Part 3: CUDA编程 

> GPU编程学习推荐这一系列博客[Link](https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89)

本部分实现了CUDA两个张量（维度分别为`batch_size, m, k`和`batch_size, k, n`）的batch内乘法。

_需要注意的是：一个block内最多只能有1024个thread，因此不能直接使用三维block去匹配三维Tensor。此外，本部分的最终代码可优化程度较高，没有利用shared memory，也没有根据张量维度设计并行模型。_