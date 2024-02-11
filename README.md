# KIM-NequIP
[![Documentation Status](https://readthedocs.org/projects/kim-torchml-port/badge/?version=latest)](https://kim-torchml-port.readthedocs.io/en/latest/?badge=latest)

## Introduction
This is a highly stripped down and edited version of original NequIP repository,
that can be used to port your existing NequIP models to KIM API. It was created against 
**NequIP 0.5.6**, and for later version it does seem to work, but not tested extensively. If 
you face any problem, please open an issue, or contact me directly.

> This is a placeholder solution before the KLIFF model builder is finished which should 
provide direct means of constructing/training compatible NequIP models.

## Caveats
This works perfectly for **three convolution layers**. If you have more or less than three layers, 
you need to modify the code accordingly. You need to edit the source files at two places:
1. `kim_nequip/scripts/convert.py`
2. `kim_nequip/nn/_graph_mixin.py`

### 1. `kim_nequip/scripts/convert.py`
This file contains the function to copy weights from the trained models to the new KIM 
compatible untrained models, and save them. You need to modify the wrapper model to
include different number layers then 3. 

You would need to modify the following lines 194, 197:
```python
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def forward(self, x, pos,
                    edge_graph0, edge_graph1, edge_graph2,  # << ADD LAYERS HERE       #194
                    contributions):
            energy = self.model(x, pos,
                                edge_graph0, edge_graph1, edge_graph2, # << ADD LAYERS HERE #197
                                contributions)
        # ...
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```
to include the layers you have in your model,
```python
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def forward(self, x, pos,
                    edge_graph0, edge_graph1, edge_graph2, edge_graph3, edge_graph4,  # << ADD LAYERS HERE       #194
                    contributions):
            energy = self.model(x, pos,
                                edge_graph0, edge_graph1, edge_graph2, edge_graph3, edge_graph4, # << ADD LAYERS HERE #197
                                contributions)
        # ...
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

### 2. `kim_nequip/nn/_graph_mixin.py`
Similarly, you need to modify the `_graph_mixin.py` file to include the layers you have in your model.
However, this process would be a bit more involved, as you need to modify the `forward` function to include
the layers you have in your model, and edit the list indexes accordingly. You also need to
modify creation of edge embeddings, and edge vectors.
The function has been annotated to guide you through the process.

As an example, if your model has 5 convolution layers, modify the `forward` function as,

```python
    def forward(self, species, coords,
                edge_index0, edge_index1, edge_index2, edge_index3, edge_index4, # << ADD LAYERS HERE
                batch):
        x = species # assignments to match the original code
        pos = coords
        contributing = batch

        # Embedding
        x_embed = self[0](x.squeeze(-1))
        x_embed = x_embed.to(dtype=pos.dtype, device=pos.device)
        h = x_embed

        # Edge embeddings
        edge_vec0, edge_sh0 = self[1](pos, edge_index0)
        edge_vec1, edge_sh1 = self[1](pos, edge_index1)
        edge_vec2, edge_sh2 = self[1](pos, edge_index2)
        edge_vec3, edge_sh3 = self[1](pos, edge_index3)
        edge_vec4, edge_sh4 = self[1](pos, edge_index4) #... << ADD LAYERS HERE

        # Radial basis functions
        edge_lengths0, edge_length_embeddings0 = self[2](edge_vec0)
        edge_lengths1, edge_length_embeddings1 = self[2](edge_vec1)
        edge_lengths2, edge_length_embeddings2 = self[2](edge_vec2)
        edge_lengths3, edge_length_embeddings3 = self[2](edge_vec3)
        edge_lengths4, edge_length_embeddings4 = self[2](edge_vec4) #... << ADD LAYERS HERE


        # Atomwise linear node feature
        h = self[3](h)

        # Conv
        # << ADD LAYERS HERE
        h = self[4](x_embed, h, edge_length_embeddings4, edge_sh4, edge_index4)
        h = self[5](x_embed, h, edge_length_embeddings3, edge_sh3, edge_index3)
        h = self[6](x_embed, h, edge_length_embeddings2, edge_sh2, edge_index2)
        h = self[7](x_embed, h, edge_length_embeddings1, edge_sh1, edge_index1)
        h = self[8](x_embed, h, edge_length_embeddings0, edge_sh0, edge_index0)
        #        ^  Edit indexes of layers accordingly (they should be in sequential order)
        #           and please note the reverse order of embeddings and edge vectors etc.
    
        # Atomwise linear node feature
        h = self[9](h)
        h = self[10](h)

        # Shift and scale
        h = self[11](x, h)

        # Sum to final energy
        h = h[contributing==0]
        energy = torch.sum(h)
        return energy
```
Such explicit modification is required to enable the parallel inference of the model.

## Installation
As the source would require considerable modification, it is recommended to install the package
in the editable mode. To install the package in the editable mode, run the following command in the
root directory of the package.

```bash
pip install -e .
```

## Usage
The install command above should have installed the `kim-nequip-port` command line tool. 
You can use this to convert existing *deployed NequIP models* to KIM API compatible models.
You would need two files for this, 
1. The deployed NequIP model, usually called `deployed.pth`
2. The NequIP configuration yaml file, usually called `config.yaml`

To convert the model, run the following command in the directory where the files are located.

```bash
kim-nequip-port config.yaml --deployed deployed.pth --out_name MY_MODEL_NAME.pt --save-kim-model
```

This will create a following files and folders in you run dir,
1. `MY_MODEL_NAME.pt` - The KIM API compatible model
2. `MY_MODEL_NAM.txt` - The model architecture information, for checking and debugging.
3. `MY_MODEL_NAME__MO_000000000000_000` - A folder containing KIM API compatible model files.

`MY_MODEL_NAME__MO_000000000000_000` is the folder that you would need to use in your KIM API
model. It should have the following files,
1. `CMakelists.txt` - The CMake file for building the model
2. `file.param` - The parameter file for the KIM API model, needed by the model driver
3. `MY_MODEL_NAME.pt` - The KIM API compatible TorchScript model

You can install this model directly ny the commnad,
```bash
kim-api-collections-management install user MY_MODEL_NAME__MO_000000000000_000
```
This would also install the model driver `TorchML__MD_173118614730_000` needed to run 
the model. Please check its KIM API for more (details)[https://openkim.org/id/MD_173118614730_000].
The following dependencies have to be met to install the model driver,
1. KIM API
2. libtorch CXX API (v1.13)
3. libtorchscatter and libtorchsparse (not the python packages, but the CXX API)

`libtorchscatter` and `libtorchsparse` CXX API can be installed from the following repositories,
1. [libtorchscatter](https://github.com/rusty1s/pytorch_scatter)
2. [libtorchsparse](https://github.com/rusty1s/pytorch_sparse)

## Inference
If the installation is successful, you can use the model driver to run the model. Below is an 
example LAMMPS input script to run the model.

```
# Initialize KIM Model
kim init MY_MODEL_NAME__MO_000000000000_000 metal

# Load data and define atom type
read_data your_data_file
kim interactions Si # for Si
mass 1 28.0855

neighbor 1.0 bin
neigh_modify every 1 delay 0 check no

# Create randome velocities and fix thermostat
velocity all create 300.0 4928459 rot yes dist gaussian
fix 1 all nvt temp 300.0 300.0 $(100.0*dt)

timestep 0.001
thermo 1
run    10
```
For ASE based evaluations, you can use the following code to run the model,
```python
import ase
from ase.io import read
from ase.calculators.kim import KIM

# Read the structure
atoms = read('your_data_file')

# Initialize the KIM model
calc = KIM('MY_MODEL_NAME__MO_000000000000_000')

# Set the calculator
atoms.set_calculator(calc)

# Run the calculation
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## License
This software is released under the original MIT License. Please see the [LICENSE](LICENSE) file for details.

## Disclaimer
This has seen limited testing in my own work, and I cannot guarantee that it will always work for you.
So please keep an eye out for any incorrect behavior, and report it to me. I will try to fix it as soon as possible.
Also, I am not responsible for any damage or loss of data caused by the use of this software.

For reporting issues, please also provide the configuration file, and a trained model,
if possible. This will help me to debug the issue more effectively. Even without the trained
models, the configuration file alone will be huge help.