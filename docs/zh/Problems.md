# Deep Go 安装问题

## 错误1

```cmd
INTEL MKL ERROR: 操作系统无法运行 %1。 mkl_intel_thread.dll.
Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll.
```

你需要将以下的文件

```cmd
~/Anaconda/envs/deepgo/Library/bin/libiomp5md.dll
~/Anaconda/envs/deepgo/Library/bin/cublas64_10.dll
~/Anaconda/envs/deepgo/Library/bin/cublasLt64_10.dll
~/Anaconda/envs/deepgo/Library/bin/cudart64_101.dll
~/Anaconda/envs/deepgo/Library/bin/cudatoolkit_config.yaml
~/Anaconda/envs/deepgo/Library/bin/cudnn64_7.dll
~/Anaconda/envs/deepgo/Library/bin/cufft64_10.dll
~/Anaconda/envs/deepgo/Library/bin/cufftw64_10.dll
~/Anaconda/envs/deepgo/Library/bin/curand64_10.dll
~/Anaconda/envs/deepgo/Library/bin/cusolver64_10.dll
~/Anaconda/envs/deepgo/Library/bin/cusparse64_10.dll
```

复制到`~/Anaconda/envs/deepgo`

注意：`deepgo`是你的Anaconda虚拟环境的名字

## 错误2

```cmd
module 'tensorflow' has no attribute 'compat'
```

你需要将`tensorflow-estimator`降级到`2.1.0`

```cmd
conda install tensorflow-estimator==2.1.0
# or
pip install tensorflow-estimator==2.1.0
```
