# OpenMM Plugin for DMFF


This is a plugin for [OpenMM](http://openmm.org) that used the trained JAX model by [DMFF](https://github.com/deepmodeling/DMFF) as an independent Force class for dynamics.
To use it, you need to create a JAX graph with DMFF with the input are the atom coordinates, box size and neighbor list, and output the energy and forces.

## Installation

### Install from source
To compile this plugin from source, three dependencies are required:
* **OpenMM, v7.7**: Could be installed with `conda install -c conda-forge openmm cudatoolkit=11.6`.  
* **[Tensorflow C API](https://www.tensorflow.org), v2.9.1**: Installed from source or download the pre-built binary from conda deepmodeling channel. Download from official website is not recommended since the issue of [C API in GPU XLA platform](https://github.com/tensorflow/tensorflow/issues/50458#issuecomment-1140817145).:
   ```shell
   # Compile from source.
   wget https://raw.githubusercontent.com/deepmodeling/deepmd-kit/master/source/install/build_tf.py
   python build_tf.py --cuda --cudnn-path {Path to cudnn installed dir} --prefix ${TENSORFLOW_LIB_INSTALLED_DIR}
   cp -r ../packages/tensorflow/tensorflow-2.9.1/tensorflow/c ${TENSORFLOW_LIB_INSTALLED_DIR}/include/tensorflow
   # Download from conda deepmodeling channel.
   conda install -c deepmodeling libtensorflow_cc=2.9.0=cuda116h4bf587c_0
   cp -r ${TENSORFLOW_SOURCE_DIR}/tensorflow/c ${TENSORFLOW_LIB_INSTALLED_DIR}/include/tensorflow
   ```
* **[cppflow](https://github.com/serizba/cppflow) header**: Since the class `cppflow::model` have no empty constructor. A small patch into the header file is required. 
  ```shell
  git clone https://github.com/serizba/cppflow.git
  cd cppflow
  git apply ${openmm_dmff_plugin_source_dir}/tests/cppflow_empty_constructor.patch
  mkdir build && cd build
  cmake .. -Dtensorflow_INCLUDE_DIRS=${LIBTENSORFLOW_INSTALLED_DIR}/include -Dtensorflow_LIBRARIES=${LIBTENSORFLOW_INSTALLED_DIR}/lib/libtensorflow.so
  ```

Compile plugin from source as following steps.

1. Clone this repository and create a directory in which to build the plugin.
   ```shell
   git clone https://github.com/dingye18/openmm_dmff_plugin.git
   cd openmm_dmff_plugin && mkdir build && cd build
   ```

2. Run `cmake` command with required parameters.
   ```shell
   cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} -DCPPFLOW_DIR=${CPPFLOW_INSTALLED_DIR} -DTENSORFLOW_DIR=${LIBTENSORFLOW_INSTALLED_DIR}
   make && make install
   make PythonInstall
   ```
   
3. Test the plugin in Python interface, reference platform.
   ```shell
   python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100
   python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA
   ```
## Usage
Add the following lines to your Python script to use the plugin.
More details can refer to the script in `python/OpenMMDMFFPlugin/tests/test_dmff_plugin_nve.py`.

```python
from OpenMMDMFFPlugin import DMFFModel
# Set up the dmff_system with the dmff_model.    
dmff_model = DMFFModel(dp_model)
dmff_model.setUnitTransformCoefficients(1, 1, 1)
dmff_system = dmff_model.createSystem(topology)
```
