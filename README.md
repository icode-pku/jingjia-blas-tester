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

### The second method(recommend)

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
cd jingjia-blas-tester
mkdir build && cd build
cmake ..
make -j4
```

4、run

```
./tester [parameters] dev-routinename
```

5、Test via script
We also wrote a shell script, which contains test cases. The shell script is in the two directories test_scripts/correctness and test_scripts/performance, which represent correctness testing and performance testing respectively.
When users use it, they only need to cd to the directory and run the following command:
```
bash test_levelxxx_xxx.sh
```

## License

Copyright (c) 2017-2022, University of Tennessee. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Tennessee nor the
      names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright holders or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.