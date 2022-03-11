## 实验目的

1. 学习基础CUDA编程
1. 完成cuda端矩阵乘法的编写，并绑定到python端使用pytorch调用

## 实验步骤

1. 编译vector add文件夹

```
nvcc -o vecAdd vecAdd.cpp vecAdd.cu
./vecAdd
```



2. 完成matrix multiplication文件夹作业

- 直接运行以下命令行编译。了解使用`develop`而不是`install`生成本地动态链接库，而不是安装至python环境之中

  ``` 
  python setup.py develop
  python main.py
  ```

- **作业仅需修改`MM.cu`文件**，完成batch matrix multiplication。要求kernel function中使用shared memory。



作业完成情况：

- 要求完成MM.cu的代码编写
- 要求能通过编译
- 要求能够通过main.py的精确度测试
- 将文件夹3.cuda programming打包成压缩包，命名为"学号 姓名 Lab1Part2"上传至canvas
