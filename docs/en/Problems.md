# Installation Problems

## Error 1

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

NOTE: `deepgo` is the name of the environment.

## Error 2

```cmd
module 'tensorflow' has no attribute 'compat'
```

You need to degradation the `tensorflow-estimator` to `2.1.0`

```cmd
conda install tensorflow-estimator==2.1.0
# or
pip install tensorflow-estimator==2.1.0
```
