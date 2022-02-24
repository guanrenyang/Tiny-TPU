# Homework 1

> Homework1 只是一个tutorial，不要提交。
## Pytorch Basics

**Just learn pytorch**

## Customed Layer 自定义算子

> 详细内容参见官网教程：https://pytorch.org/docs/stable/notes/extending.html#extending-autograd

利用`torch.autograd.Function`和`torch.nn.Module` 自定义算子的`forward()`, `backward()`操作:

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

## CPP Binding

> CPP Binding官网教程：https://pybind11.readthedocs.io/en/stable/

本次作业的CPP Binding并非使用C++定义模型，只是使用C++定义函数以供Pytorch调用。

[`setup.py`](https://github.com/guanrenyang/AI3615-AI-Chip-Design/blob/main/hw1/2.%20customed%20layer%20%2B%20cpp%20binding/2.%20cpp%20binding/setup.py)的作用是 _安装_ `.cpp`文件，将一个`.cpp`文件打包为Python Module。
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
