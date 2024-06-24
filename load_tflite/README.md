# compile tensorflow lite library (.so) and configure lib and include in of your project
step 0:
  download: https://github.com/bazelbuild/bazelisk/releases
step 1:
  chmod +x /data_ws/Data_1/tinghao/software/bazelisk-linux-amd64
step 2:
  sudo mv /path/to/bazelisk-linux-amd64 /usr/local/bin/bazel
  which bazel (output: /usr/local/bin/bazel)
step 3:
  git clone https://github.com/tensorflow/tensorflow.git
step 4:
  cd tensorflow
step 5:
  ./configure
step 6:
  bazel build --cxxopt='--std=c++17' //tensorflow/lite:libtensorflowlite.so
step 7:
  sudo cp ./bazel-bin/tensorflow/lite/libtensorflowlite.so /path/to/your/project/lib/
step 8:
  sudo mkdir /path/to/your/project/include/tensorflow
step 9:
  sudo cp -r ./tensorflow/lite /path/to/your/project/include/tensorflow
step 10:
  sudo cp -r ./tensorflow/core /path/to/your/project/include/tensorflow
step 11:
  sudo -r ./bazel-bin/external/flatbuffers/_virtual_includes/flatbuffers/flatbuffers /path/to/your/project/include

# Build (compile)

```
cd load_tflite
mkdir build
cd build
cmake ..
make -j$(nproc) 
```

# Run
```
cd load_tflite/build
./TFLiteCheck ../model/classification/mobilenet_v1_1.0_224_quant.tflite ../model/classification/labels_mobilenet_quant_v1_224.txt ../data/classification_example.jpg
```

# reference pages
https://zhuanlan.zhihu.com/p/462494086