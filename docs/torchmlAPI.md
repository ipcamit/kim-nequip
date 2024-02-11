# Supported models

TorchML model driver does not require any explicit restriction on the models.
It supports **all** models that can be converted to TorchScript using `torch.jit.script` module.
As TorchML model driver uses the CXX API of libtorch, it can run every model in principle.
But for interference with KIM API, the model need to accept inputs in a certain format and 
return the energy and forces, or just the energy.

## Required Signatures
To make any model compatible with TorchML model driver, you just need to ensure two things
1. The model's `forward` function should accept the inputs in the format given below
2. The model should be convertible to TorchScript using `torch.jit.script` module.

The following 3 models are supported by the TorchML model driver:

|Models|Signature|
|:-----|:--------|
|Generic, all encompassing model |`forward(self, species, coords, n_neigh, nlist, contributing)`|
|Descriptor based models | `forward(self, descriptor)`|
|Graph neural networks | `forward(self, species, coords, graph_layer1, graph_layer2,..., contributing)`|

You can see the example implementations of all three portable model kinds in
the [example portable models](https://github.com/ipcamit/colabfit-portable-models) repository.
The models are named as follows:

|Models| Folder in `colabfit-portable-models`                                                              |
|:-----|:--------------------------------------------------------------------------------------------------|
|Generic, all encompassing model | `TorchMLModel1_SW`, a Stillinger-Weber model implemented in pytorch to highlight the flexibility. |
|Descriptor based models | `TorchMLModel2_Desc`, is a simple Symmetry functions based neural network                         |
|Graph neural networks | `TorchMLModel3_Graph`, implements a generic 3 layer, GNN                                          |

## Basic porting steps
To port any existing model to TorchML model driver, easiest way is to write a dummy model,
which adheres to the KIM TorchML model driver signature. Let's call this model `AdaptorModule`
Then wrap the existing model in the wrapping `AdaptorModule`. For this you would need to write 
the code to convert the KIM API inputs as per your model's requirements. Following this
you can convert the model to TorchScript using `torch.jit.script` module. Now bundle it 
with the parameter file and you are ready to go.

## Basic model file structure
The basic KIM API portable model has a CMakeLists.txt file, a parameter file. In case
of TorchML model you would also need the TorchScript model file. In case of the ML driver
the model is a "parameter" as well.

:::{tip}
Following extensions are must for all files, `.pt` for TorchScript model file, `.param` for parameter file, and `.dat` for descriptor file.
:::

The portable model has the directory structure as follows:
```{bash}
MY_MODEL_NAME__MO_000000000000_000
├── CMakeLists.txt
├── descriptor.dat
├── file.param
└── MyTorchScriptFile.pt
```

and CMakeLists.txt file usually looks like this:
```{cmake}
cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

kim_api_items_setup_before_project(ITEM_TYPE "portableModel")
project(YOUR_MODEL_NAME__MO_000000000000_000) # < Change this to your model name
kim_api_items_setup_after_project(ITEM_TYPE "portableModel")

add_kim_api_model_library(
  NAME            ${PROJECT_NAME}
  DRIVER_NAME     "TorchML__MD_173118614730_000" # < Model driver ID
  PARAMETER_FILES "file.param" "MyTorchScriptFile.pt" "descriptor.dat" # < Add the model files here
  )
```
`MyTorchScriptFile.pt` is the TorchScript model file, and `file.param` is the parameter file.

The parameter file contains the information about the model, in following format,
```
# Num of elements: how many species the model supports and what are they
2
Si Al

# Preprocessing layer": what kind of model it is (Descriptor, Graph, or None) case sensitive
Graph

# Cutoff Distance (Angstroms), if it is a GNN, then number of GNN layers in next line
4.0
3

# TorchScript model file name
MY_TORCH_MODEL.pt

# Does the model return energy and forces?
# False means only energy is returned, and the ML driver would calculate the forces
True

# Number of Inputs to the model
# for generic models, it is 5: species, coords, n_neigh, nlist, contributing
# for Descriptor based models, it is 1
# for GNN based models, it is 1 (species) + 1 (coords) + 1 (contributing) + n (conv layers)
# 3 conv layers  
6

# Descriptor: name of the descriptor to use
# Ignored for GNN and generic models
# Value: SymmetryFunctions and Bispectrum for now
None
```

