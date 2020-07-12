<div align='right'>
  Language:
  US
  <a title="Chinese" href="docs/zh-CN/README.md">CN</a>
</div>

# Deep Go

Deep Go is an framework for machine learning and deep learning.

<p align='center'>
<a href="docs/more.md"><img src="https://img.shields.io/badge/version-1.0.0a-yellow.svg"></a>
<a href="docs/more.md"><img src="https://img.shields.io/badge/TensorFlow-=2.1.0-green.svg"></a>
<a href="docs/more.md"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg"></a>
</p>

## Installation

1. Clone Deep Go into your computer. You can download the zip file or use `GitHub Desktop` to clone it.
2. Add Deep Go folder to your `PYTHONHOME`
3. Install the requirements by using `Anaconda`. New environment is Recommended.

```cmd
conda create -n deepgo python=3.7.7 --yes
conda activate deepgo
conda install tensorflow-gpu=2.1.0 --yes
```

`deepgo` is the name of the environment, you can use your favorite name.

### Error handling

#### Error 1

```cmd
INTEL MKL ERROR: 操作系统无法运行 %1。 mkl_intel_thread.dll.
Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll.
```

You need to copy these file into this path: `~/Anaconda/envs/deepgo`

* `~/Anaconda/envs/deepgo/Library/bin/libiomp5md.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cublas64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cublasLt64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cudart64_101.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cudatoolkit_config.yaml`
* `~/Anaconda/envs/deepgo/Library/bin/cudnn64_7.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cufft64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cufftw64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/curand64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cusolver64_10.dll`
* `~/Anaconda/envs/deepgo/Library/bin/cusparse64_10.dll`

#### Error 2

```cmd
module 'tensorflow' has no attribute 'compat'
```

You need to degradation the `tensorflow-estimator` to `2.1.0`

```cmd
pip install tensorflow-estimator==2.1.0
```

## Usage

Deep Go is very easy to use

```python
import deepgo as dg
```

## API

See [Deep Go API](docs/api/README.md)

## Code Standard

See [Deep Go Code Standard](docs/CodeStandard.md)

## Contributor

* York Su
