## 实验目的

1. 学习自定义算子
2. 了解cpp代码到python端的binding



## 实验步骤

### 1. 自定义算子

1. 阅读customed_layer文件夹中的代码，了解利用`torch.autograd.Function`和`torch.nn.Module` 自定义算子的前向反向操作。
2. 学习编写`torch.autograd.Function`，注意如何在forward函数中把某个Tensor暂存，以用在backward函数中。
3. 阅读`torch.autograd.Function`的官网教程：https://pytorch.org/docs/stable/notes/extending.html#extending-autograd。



### 2. C++ binding

1. 阅读cpp binding文件夹中的代码，完成以下步骤，确认程序正常运行。

   ```
   python setup.py install
   python cpp_binding_example.py
   ```

   

2. 注意mylinear.cpp中`PYBIND11_MODULE` 的部分。模仿mylinear.cpp，**编写一个c++的函数`cppprint`**，使得python端能够调用c++的`std::cout`完成打印输出。

   

3. 【扩展阅读】如对c++到python端的binding感兴趣，可以查阅pybind11的官网https://pybind11.readthedocs.io/en/stable/ 学习，本实验暂不做额外要求。
