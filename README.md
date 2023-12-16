# cuda_testbed_ubuntu
C++ CUDA Testbed  (Ubuntu)

## Prerequisites

GCC and NINJA and CMAKE must be installed

```
sudo apt-get install gcc
sudo snap install cmake --classic
sudo apt-get install git
sudo apt-get install ninja-build

## Build


### Prepare
```
mkdir build
cd build
cmake -G Ninja ..
```

### Build

```
ninja
```

### Run

./testbed_cpp20


