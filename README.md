* * *

[TOC]

* * *

About
--------------------------------------------------------------------------------

The jingjia-blas-tester evaluation software package is used to provide a basis for the establishment of a general computing library evaluation system for Jingjia Micro's next-generation GPU. The content includes establishing API function completeness evaluation standards and performance evaluation standards for the cuBlas library.
     When comparing correctness, for computing APIs, we use the results of cblas and cublas for comparison. For the correctness test of the helper API, we make correctness judgments based on the calculation results and return status codes.
     When conducting API performance testing, we mainly test time, Gflops, and Gbytes.

​     When testing for correctness, the results of the CUBLAS API are compared with the results of the Cblas calculation library. Due to the diversity of host-side CPU types, our evaluation framework supports multiple Cblas versions, including mkl_cblas, openblas, original cblas, etc. Evaluation users can choose the Cblas version according to their own hardware environment. 

* * *

Compile
--------------------------------------------------------------------------------

​        Before compiling the software source code package, please make sure there are cblas and lapack computing libraries in the environment, such as cblas, mkl_cblas or openblas. At the same time, you need to ensure that you have a CUDA11.8 environment. There are two recommend ways to build the environment：

It is recommended to use **docker** for compilation

### The first method

1、Pull docker image

```
docker pull nvcr.io/nvidia/pytorch:23.07-py3
```

url:url: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

2、you can build this project.

```
cd jingjai-blas-tester
mkdir build && cd build
cmake ..
make -j4
```

### The second method

1、Pull docker image

```
docker pull thebloke/cuda11.8.0-ubuntu22.04-pytorch
```

2、Environment setup

```
apt-get update
apt install libblas3 libblas-dev
apt-get install libblas-dev liblapack-dev
apt-get install libatlas-base-dev
apt install libopenblas-dev
```

3、make

```
cd jingjai-blas-tester
mkdir build && cd build
cmake ..
make -j4
```

4、run

```
./tester [parameters] routine
```

