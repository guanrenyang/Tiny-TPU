# AI3615 AI Chip Design

## Introduction

上海交通大学AI3615人工智能芯片设计课程项目。

Course project of AI3615 AI chip design, Shanghai Jiao Tong University(SJTU).

## Contents

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
├── LICENSE
└── README.md

```

