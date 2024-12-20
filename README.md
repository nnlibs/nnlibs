# nn_lib

#### 介绍

神经网络 C++ 实现

CPU 侧加速使用到了如下方式进行实现:

- [x] OpenMP
- [x] SIMD
- [x] OpenBLAS

目前已经实现的算子:

- [x] relu
- [x] sigmod
- [x] conv_2d
- [x] linear
- [x] max_pool_2d
- [x] cross_entropy
- [ ] self_attention
- [ ] multi_head_attention
- [ ] embedding

#### 软件架构

软件架构说明 TODO

#### 安装教程

项目安装方式 TODO

#### 使用说明

编译项目

```sh
mkdir build
cd build
cmake ..
make -j8
```

1. 分类模型训练与测试

首先下载模型训练与测试需要使用的数据集

```sh
wget 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
#解压
tar xzf cifar-10-binary.tar.gz
```

2. 进行训练以及测试

```sh
./tests/ClaTest <train_file_prefix>
```

#### 待执行

- [ ] 加入更多算子
- [ ] 实现小型的 NLP 网络
- [ ] 支持模型训练参数的导出
- [ ] CUDA 加速模型训练
