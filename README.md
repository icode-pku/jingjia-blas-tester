* * *

[TOC]

* * *

About
--------------------------------------------------------------------------------

The jingjia-blas-tester evaluation software package is used to provide a basis for the establishment of a general computing library evaluation system for Jingjia Micro's next-generation GPU. The content includes establishing API function completeness evaluation standards and performance evaluation standards for the cuBlas library.
     When comparing correctness, for computing APIs, we use the results of cblas and cublas for comparison. For the correctness test of the helper API, we make correctness judgments based on the calculation results and return status codes.
     When conducting API performance testing, we mainly test time, Gflops, and Gbytes.

* * *

Installation 
--------------------------------------------------------------------------------

Before installing the software source code package, please make sure there are cblas and lapack computing libraries in the environment, such as cblas, mkl_cbals or openblas. At the same time, you need to ensure that you have a CUDA11.8 environment.

It is recommended to use **docker** for compilation

#### 1、Pull docker image

```
docker pull thebloke/cuda11.8.0-ubuntu22.04-pytorch
```

#### 2、Environment setup

```
apt-get update
apt install libblas3 libblas-dev
apt-get install libblas-dev liblapack-dev
apt-get install libatlas-base-dev
apt install libopenblas-dev
```

#### 3、make

```
cd jingjai-blas-tester
mkdir build && cd build
cmake ..
make -j4
```

#### 4、run

```
./tester [parameters] routine
```

