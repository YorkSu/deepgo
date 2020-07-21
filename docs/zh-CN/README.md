<div align='right'>
  语言:
  <a title="英文" href="../../README.md">英文</a>
  中文
</div>

# Deep Go

Deep Go 是一个机器学习和深度学习的框架。

<p align='center'>
<a href="docs/more.md"><img src="https://img.shields.io/badge/version-1.0.0a-yellow.svg"></a>
<a href="docs/more.md"><img src="https://img.shields.io/badge/TensorFlow-=2.1.0-green.svg"></a>
<a href="docs/more.md"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg"></a>
</p>

## 安装

1. 将Deep Go克隆到本地，下载zip或使用`GitHub Desktop`进行克隆
2. 将Deep Go文件夹添加到系统变量`PYTHONPATH`
3. 使用Anaconda安装依赖，推荐创建新的虚拟环境。

```cmd
conda create -n deepgo python=3.7.7 --yes
conda activate deepgo
conda install tensorflow-gpu=2.1.0 --yes
```

`deepgo` 是虚拟环境的名字，你可以使用你喜欢的名字。

如果你安装之后遇到了问题，请查看[安装问题](docs/Problems.md)

## 使用

Deep Go 的使用非常简单

```python
import deepgo as dp
```

## API

参见 [Deep Go API](docs/api/README.md)

## 代码规范

参见 [Deep Go 代码规范](docs/CodeStandard.md)

## 贡献者

* York Su
