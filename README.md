# OpenMM Plugin for DMFF


This is a plugin for [OpenMM](http://openmm.org) that used the trained JAX model by [DMFF](https://github.com/deepmodeling/DMFF) as an independent Force class for dynamics.
To use it, you need to create a JAX graph with DMFF with the input are the atom coordinates, box size and neighbor list, and output the energy and forces.

## Installation

### Install from source
To compile this plugin from source, three dependencies are required:
* **OpenMM, v7.7**: Could be installed with `conda install -c conda-forge openmm cudatoolkit=11.7`.  
* **[Libtensorflow](https://www.tensorflow.org/install/lang_c), v2.11.0**: Installed as the following steps:
   ```shell
   FILENAME=libtensorflow-{cpu/gpu}-linux-x86_64-2.11.0.tar.gz
   wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
   tar -C ${LIBTENSORFLOW_INSTALLED_DIR} -xzf ${FILENAME}
   export LD_LIBRARY_PATH=${LIBTENSORFLOW_INSTALLED_DIR}/lib:$LD_LIBRARY_PATH
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
