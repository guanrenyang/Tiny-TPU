## 实验目的

构建pytorch环境，学习使用pytorch构建模型



## 实验步骤

### 1. 安装pytorch

1. 安装anaconda， 教程：https://docs.anaconda.com/anaconda/install/linux/#installation

2. 安装python 3.7.6，

   ```
   conda create -n AIChip python=3.7.6
   conda activate AIChip
   ```

   

3. 安装pytorch 1.7.0，教程：https://pytorch.org/get-started/previous-versions/

   ```
   pip install torch==1.7.0+cu92 torchvision==0.8.0+cu92 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
   
   ```

   本次实验暂不使用GPU，若没有GPU资源可以安装pytorch的cpu版本，详见教程链接



### 2. 了解pytorch的基础

1. 阅读https://pytorch.org/tutorials/beginner/introyt.html 的

   - 1. [Introduction to PyTorch](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)、

   - 2. [Introduction to PyTorch Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
   - 4. [Building Models with PyTorch](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html) 
2. 学习pytorch的Tensor构建
3. 学习pytorch的模型构建
4. 参考 mnist_example.py，了解pytorch的基本模型训练方式



### 3. 参考 mnist_example.py，完成以下内容：

- 自己设计一个有5个卷积层和两个全连接层的模型
- 使用pytorch编写该模型
- 修改 mnist_example.py，使得模型能够运行

