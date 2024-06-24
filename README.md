# General
if you want to load tensorRT engine file, please go to /load_engine

if you want to load onnx file, please go to /load_onnx

# Convert onnx to engine file

```
./trtexec --onnx=&{path_to_onnx}/xxx.onnx  --saveEngine=${path_to_output}/xxxxx.trt 

```

or

```
./trtexec --onnx=&{path_to_onnx}/xxx.onnx  --saveEngine=${path_to_output}/xxxxx.engine
```
or

```
./trtexec --onnx=&{path_to_onnx}/xxx.onnx  --saveEngine=${path_to_output}/xxxxx.plan
```

you could find trtexec in /usr/src/tensorrt/bin

# How to modify the example

## 1. change tensorname of input and output
in load_eingine.cpp changes the following code accordingly:

```
  params.inputTensorNames.push_back("your_input_tensor_name");
  params.outputTensorNames.push_back("your_output_tensor_name");
```

## 2. change the dimensions of the input of network
change the following height and width of the input image according to your network in load_eingine.cpp

```
  inputH = mInputDims.d[index_of_input_image_height];
  inputW = mInputDims.d[index_of_input_image_width];
```


tips: this name shall be the same as your network
# Build (compile)

```
cd load_engine
mkdir build
cd build
cmake ..
make -j$(nproc) 
```

# Run
```
cd load_engine/build
./dynamic_gesture --datadir ../model
```

# measure GPU consumption
```
nsys profile --sample cpu ./dynamic_gesture --datadir ../model
```

# GFLOPS via taskset, htop
```
taskset -cp 0 pid
htop
```

# install polygraph tool to detect output precision error between .onnx and .trt
method 1: 
```
1. conda create -n env_name python=3.10
2. download repo from https://github.com/NVIDIA/TensorRT/tree/release/8.5  (you need to select proper version)
3. cd ./tools/Polygraph
4. pip install onnx==1.12.0 & onnxruntime==1.12.1
5. python setup.py install
5. optional: pip install colored 
6. pip install tensorrt==8.5.3.1 (same version as 2.)
```

method 2:
```
1. conda create -n env_name python=3.10
2. pip install onnx==1.12.0 & onnxruntime==1.12.1
3. pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
4. pip install tensorrt==x.x.x.x (you need to select proper version)
```

# use polygraph tool to detect output precision error between .onnx and .trt
```
polygraphy run /data_ws/Data_1/tinghao/hand_pipe/load_engine/model/hand_detection_0821.onnx --trt --onnxrt --trt-outputs mark all --onnx-outputs mark all --atol 1e-3 --rtol 1e-3 --fail-fast --val-range [0,1]
```

# compile .cu file to .so / .a (pre-request: cudatoolkit installed)
step by step:
  0. confirm wether cudatoolkit install normally or not: 
    ```
    nvcc -V
    ```
  1. compile .cu to .so (動態庫):
    ```
    nvcc -Xcompiler "-fPIC" path/to/your/xxx.cu -c -I path/to/your/TensorRT-x.x.x.x/include
    nvcc -shared path/to/your/xxx.o -o path/to/your/libxxx.so
    ```
  2. convert .onnx to .trt
    ```
    path/to/your/trtexec --onnx=path/to/your/onnx_model.onnx --saveEngine=path/to/your/trt_model.trt --plugins=path/to/your/libxxx.so
    ```
  3. if your TensorRT version < 8.0.1.6 GA, then: 
    add "initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");" to 1st line in function build(), 
    pls refer to this url: https://blog.csdn.net/HW140701/article/details/120377483
  4. compile and run your code
addition information:
  compile .cu to .a (靜態庫):
    ```
    nvcc -lib xxx.cu -o libxxx.a
    ```
example (compile .cu to .so on X86 platform: tinghao@10.80.104.156):
  a. go to https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/old  (Nivida Official)
  b. download ./plugins
  c. compile ./plugins/ScatterND.cu -> libScatterND.so according to aboving steps
    1. nvcc -Xcompiler "-fPIC" ./plugins/ScatterND.cu -c -I /opt/TensorRT-8.5.3.1/include
    2. nvcc -shared ScatterND.o -o libScatterND.so
    3. /opt/TensorRT-8.5.3.1/bin/trtexec --onnx=/data_ws/Data_1/tinghao/trt-samples-for-hackathon-cn-master/old/hand_detection_0821.onnx --saveEngine=/data_ws/Data_1/tinghao/trt-samples-for-hackathon-cn-master/old/hand_detection_0821.trt --plugins=/data_ws/Data_1/tinghao/trt-samples-for-hackathon-cn-master/old/ScatterND.so
    4. pls refer to tinghao@10.80.104.156: /data_ws/Data_1/tinghao/hand_pipe/load_engine/Net.hpp: line 83

# trtexec for transform onnx model format to .engine/.trt/.plan model format  via TensorRT [ trtexec method on tinghao@10.80.104.156]
  1. float32 model.trt:
  /opt/TensorRT-8.5.3.1/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt

  [or]

  /opt/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt
  2. int8 model.trt:
  /opt/TensorRT-8.5.3.1/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt --batch=1 --int8 --calib=path/to/calib_file.cache
  [or]
  /opt/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt --batch=1 --int8 --calib=path/to/calib_file.cache
  3. float16 model.trt:
  /opt/TensorRT-8.5.3.1/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt --fp16
  [or]
  /opt/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/bin/trtexec --onnx=path/to/model.onnx --saveEngine=path/to/save/model.trt --fp16

# load_engine.cpp中的宏定義：
#define METHOD 2  // 0: camera 1: video 2: image
#define REMOTE 1  // 0: not remote operation (by ssh), 1: remote operation (by ssh)
#define PLATFORM 0  // 0: X86 PC, 1: Nvidia Jetson

# compile and use 'eigen'
step 1:
  download eigen-x.x.x source code: http://eigen.tuxfamily.org/index.php?title=Main_Page
step 2:
  cd eigen-x.x.x
step 3:
  mkdir build
step 4:
  cd build
step 5:
  cmake ..
step 6:
  sudo make install
step 7:
  add to CmakeLists.txt: include_directories(${CMAKE_CURRENT_SOURCE_DIR}/relative/path/to/eigen-x.x.x)
