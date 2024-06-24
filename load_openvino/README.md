# download [openvino runtime] and configure lib and include in of your project
step 1:
  https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html?ENVIRONMENT=RUNTIME&OP_SYSTEM=LINUX&VERSION=v_2023_0_1&DISTRIBUTION=ARCHIVE

[note]: please select [OpenVino Archives], do not compile OpenVino by yourself (many weird problems)

step 2:
  tar -xzvf /path/to/your/openvino_toolkit_package.tgz

step 3:
  sudo cp -r /data_ws/Data_1/tinghao/software/openvino_toolkit_package/runtime/lib/intel64 /path/to/your/project/lib

step 4:
  sudo cp -r /data_ws/Data_1/tinghao/software/openvino_toolkit_package/runtime/include/ie /path/to/your/project/include

  sudo cp -r /data_ws/Data_1/tinghao/software/openvino_toolkit_package/runtime/include/ngraph /path/to/your/project/include

  sudo cp -r /data_ws/Data_1/tinghao/software/openvino_toolkit_package/runtime/include/openvino /path/to/your/project/include

step 5:
  sudo apt-get install libpugixml1v5

  sudo apt-get install libpugixml-dev

# download [openvino dev]
step 1:
  https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=LINUX&VERSION=v_2023_0_2&DISTRIBUTION=PIP&FRAMEWORK=ONNX%2CPYTORCH%2CTENSORFLOW_2

step 2:
  conda create -n openvino_dev_x_x_x python==3.x

step 3:
  conda activate openvino_dev_x_x_x

step 4:
  pip install openvino-dev[framework1,framework2,framework3, ...]==x.x.x
  example: pip install openvino-dev[ONNX,pytorch,tensorflow2]==2023.0.2

step 5:
  git clone https://github.com/openvinotoolkit/open_model_zoo.git

step 6:
  Installing Python Model API package:
  pip install <open_model_zoo_dir>/demos/common/python

step 7:
  To verify the package is installed, you might use the following command:
  python -c "from openvino.model_zoo import model_api"

step 8:
  Install Model API Adapters:
  pip install <omz_dir>/demos/common/python[ovms]

[Note]: for step 6, step 7, step 8, you can refer to following url for details
https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/openvino/model_zoo/model_api/README.md#installing-python

# Build (compile)
```
cd load_openvino
mkdir build
cd build
cmake ..
make -j$(nproc) 
```

# Run
[C++]:
```
cd load_openvino/build
./OpenVinoCheck
```

[python]:
```
conda activate openvino_x_x_x
cd load_openvino
python ./main.py
```

# the commands that we always use
```
/home/tinghao/anaconda3/envs/openvino_2023_0_2/bin/mo -h

/home/tinghao/anaconda3/envs/openvino_2023_0_2/bin/omz_downloader -h

/home/tinghao/anaconda3/envs/openvino_2023_0_2/bin/omz_converter -h
```

# downlaod model from openvino model zoo ()
step 1:
  conda activate openvino_x_x_x

step 2:
  /path/to/openvino_x_x_x/bin/omz_downloader --print_all

step 3:
  /path/to/openvino_x_x_x/bin/omz_downloader --name mobilenet-ssd --output_dir /path/to/your/parent_dir_of_public_folder (example: /path/to/your/parent_dir_of_public_folder --> /data_ws/Data_1/tinghao/hand_pipe_2/load_openvino/model, 'public' folder is a subdirectory here)

# convert model to openvino IR format
step 1:
  conda activate openvino_x_x_x

step 2:
  cd /path/to/your/parent_dir_of_public_folder

step 2:
  /path/to/openvino_x_x_x/bin/omz_converter --name mobilenet-ssd --output_dir /path/to/your/parent_dir_of_public_folder

# refernece
  https://zhuanlan.zhihu.com/p/603740365