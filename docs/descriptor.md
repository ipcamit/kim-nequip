# Descriptor Models
Descriptor based models are inherently local, and hence one of the easiest to port.
Most descriptor models can be used out of the box without any modifications. 
The only requirement is to convert the model to TorchScript using `torch.jit.script` module. 
You would need to provide the parameter file, and the descriptor file to the model driver.
The model driver will take care of the rest.

The KIM model directory structure for a descriptor based model would look like this:
```shell
MY_MODEL_NAME__MO_000000000000_000
├── CMakeLists.txt
├── descriptor.dat
├── file.param
└── MyTorchScriptFile.pt
```
`MyTorchScriptFile.pt` is the TorchScript model file, and `file.param` is the parameter file.
The descriptor hyperparameters are stored in the `descriptor.dat` file.

:::{tip}
Following extensions are must for files, `.pt` for TorchScript model file, `.param` for parameter file, and `.dat` for descriptor file.
:::

## Limitations
Biggest limitation of the descriptor models is the descriptor computation. 
TorchML model driver utilizes the [`libdescriptor`](https://libdescriptor.readthedocs.io/en/latest/) library to calculate the forces from 
the gradients of the energy with respect to the descriptors. Hence, you are required to
use libdescriptor for computing the descriptors. 

This means any descriptor that is not supported by libdescriptor is not supported by the model driver.
So if you have some custom descriptor, you would need to contact us so that we can include 
your descriptor in the libdescriptor library.

:::{important}
Currently only Behler-Parrinello Symmetry Functions, and the Bispectrum are supported by the model driver.
But SOAP and Xi would be available soon.
:::

[Example descriptor.dat file](#example-descriptor)
