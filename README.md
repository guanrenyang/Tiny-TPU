# AI3615 AI Chip Design

## Introduction

上海交通大学AI3615人工智能芯片设计课程项目。

Course project of AI3615 AI chip design, Shanghai Jiao Tong University(SJTU).

## Contents

### Homework 1: Parallel Programming

1. **Customed layer in pytorch**
2. **CPP binding**
3. **Cuda programming——Matrix Multiplication Kernel**

### Homework 2: Tiny TPU with Systolic Array

**Architecture**

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/img/arch.png" alt="arch" style="zoom: 67%;" />

**Systolic Array Design**

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/img/cycle-4.png" alt="cycle-4" style="zoom: 40%;" />



##  Directory Structure

```
├── homework1					//第一次作业
│   ├── 1. pytorch basics			//第一次作业 Part 1：pytorch基础
│   │   ├── mnist_example.py
│   │   └── README.md
│   ├── 2. customed layer + cpp binding	//第一次作业 Part 2：自定义算子 + C++ binding
│   │   ├── 1. customed_layer			//自定义算子
│   │   │   ├── customed_layer_example.py
│   │   │   └── mylinear.py
│   │   ├── 2. cpp binding			//C++ binding
│   │   │   ├── cpp_binding_example.py
│   │   │   ├── mylinear.cpp			//要打包的C++文件
│   │   │   ├── mylinear.py			//调用mylinear.cpp的文件
│   │   │   └── setup.py			//用户打包.cpp文件
│   │   └── README.md
│   ├── 3. cuda programming			//第一次作业 Part 3：CUDA编程
│   │   ├── 1. vector add			//样例代码：向量相加
│   │   │   ├── vecAdd
│   │   │   ├── vecAdd.cpp
│   │   │   └── vecAdd.cu
│   │   ├── 2. matrix multiplication		//作业代码：矩阵乘法
│   │   │   ├── csrc
│   │   │   │   ├── bind_torch.cpp
│   │   │   │   └── MM.cu
│   │   │   ├── main.py
│   │   │   └── setup.py
│   │   └── README.md
│   └── README.md
├── homework2				//第二次作业: Tiny-TPU
│   ├── AI Chip Design Lab2.pdf
│   ├── data_files
│   │   ├── instructions.dat
│   │   └── shared_memory_contents.dat
│   ├── Lab2说明.pdf
│   ├── pictures
│   ├── README.md
│   ├── src
│   │   ├── controller.sv
│   │   ├── decoder.sv
│   │   ├── elementwise_array.sv
│   │   ├── elementwise_unit.sv
│   │   ├── input_buffer.sv
│   │   ├── instruction_buffer.sv
│   │   ├── PE_array.sv
│   │   ├── PE.sv
│   │   ├── shared_memory.sv
│   │   ├── top.sv
│   │   └── weight_buffer.sv
│   └── test_bench
│       └── tb_top.sv
├── LICENSE
└── README.md


```

