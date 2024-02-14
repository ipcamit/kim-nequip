# Graph Neural Networks

Converting a GNN to TorchML format is most involved and presents several complications.
The biggest of them all the message passing in a periodic system. It is complicated
for two reasons, first the cyclic graph convolutions and second the periodic boundary
conditions.

## Cyclic graph convolutions

Most conventional GNN use convolutional layers that pass messages between atoms, and
they look like,

```python
for i in range(n_conv):
    h = graph_conv(h, edge_index)
```

where `h` is the atom embeddings, and `edge_index` defines the edge connections between
atoms. The `graph_conv` function is a convolutional layer that takes the atom embeddings
and the edge connections, and returns the updated atom embeddings.
Here the input of one model layer is the output of the previous layer, and this in effect
propagate the influence sphere/information from atoms far beyond the cutoff distance.
This is at odds with KIM API design philosophy as this influence now has a "non-local"
component. The feature of atom `i` is influenced by the feature of atom `j` and feature of
atom `j` is influenced by the feature of atom `k`, but atom `k` is beyond the cutoff distance
from atom `i`.

## Staged graph convolutions

To mitigate such problems in conventional interatomic potentials, the
KIM API provides two distances that user can define, the cutoff distance and the influence
distance. Therefore, in current implementation of the TorchML model driver, the GNN model
must use the influence distance to limit the influence sphere of the model. The influence
distance for a GNN is defined as {math}`r_{infl} = n_{conv} \times r_{cutoff}` (see figure below
for an example of {math}`n_{conv} = 2`).

```{figure} _static/infl_dist.png
:width: 400px
:alt: infl and cutoff

Influence distance and cutoff distances for a GNN model with 2 layers.
```

This poses challenges in constructing the model, as the model must now use different set of
superseding edge graphs to correctly give the multiple message passings like above.
In essence, the convolutions must look like,

```python
h = graph_conv(h, edge_index1) # convolve with r_cutoff * 2 sphere of atoms, r_infl
h = graph_conv(h, edge_index0) # final convolution with r_cutoff sphere of atoms
```

Please pay attention to the inverse order of convolution, where the first convolution is
over the entire sphere of influence, and the last convolution is over the r_cutoff sphere.
Here, `edge_index1` is a graph for all the atoms in the influence sphere thus first convolution
operation (gray circle, labelled Graph Conv 1, on the left in the figure below) calculates
feature vector for all the atoms in the cutoff distance sphere, and `edge_index0` is a
graph for all the atoms in the cutoff distance sphere (red circle on the right in the figure below).

```{figure} _static/conv.png
:width: 400px
:alt: staged graph

Staged graph convolutions.
```

:::{tip}
Please note the image of red central atom on the left gray circle, which indicates that
red atom is within the cutoff distance of the white atom, hence for first convolution
it contributes to the feature of white atom. In second convolution this summed up feature
contributes to the feature of the red atom. Thus giving equivalent representation as the
cyclic graph convolutions.
:::

## Periodic boundary conditions

The second problem is the periodic boundary conditions. The GNN model must be able to
compute correct edge vectors and edge distances compensated correctly for the lattice vectors.
A trivial implementation of it looks like,

```python
temp_r_ij = r_i - r_j
if all(temp_r_ij @ lattice_vectors < box_dims):
    r_ij = temp_r_ij
else:
    rij = temp_r_ij - mask( temp_r_ij @ lattice_vectors) *\
    lattice_vectors
    h_i = phi_h(h_i, h_j) # compute message
```

This approach requires `lattice_vectors` and `box_dims` to be passed to the model, which is
not possible in the KIM API. The model must be able to compute the edge vectors and edge distances
correctly without the lattice vectors. Turns out, the staged graph approach helps here too.
As it is purely based on influence distance, all the atoms it needs to compute the edge vectors
are correctly unwrapped to compute the edge vectors as,

```python
r_ij = r_i - r_j
if (distance(r_ij) < cutoff:
    h_i = phi_h(h_i, h_j) # compute message
```

Thus staged graph convolutions are enough to make GNN base interatomic potentials local, and
thus compatible with the KIM API. Added major benefit of staged graph convolutions is that

## Generating staged graphs

KLIFF provides with tools to generate these staged graphs,

```python
from kliff.transforms.configuration_transforms.graphs import KIMDriverGraph
graph_generator = KIMDriverGraph(species=["Si"], cutoff=4.0, n_layers=2)
graph = graph_generator.forward(configuration)
```

where `n_layers` is the number of layers in the GNN model. This should generate the edge
graphs for the GNN model.

```shell
$ print(graph)
# PyGGraph(energy= ... edge_index0=[2, 1478], edge_index1=[2, 4622])
```

As you can see the `edge_index1` is the larger graph, and would convoluted first, followed
by smalled `edge_index0` graph.

:::{tip}
The `KIMDriverGraph` is a part of `kliff` package, and depends upon Pytorch Geometric library.
For GNN support TorchML driver currently only supports the Pytorch Geometric library.
:::

## GNN model signature

As described in the [API](#signature-target) section, the GNN model must have a signature

```python
def forward(self, species, coords, edge_index0, edge_index1, edge_index2, ..., contributions)
```

where,

1. `species` is a vector of atomic indices (see [species](#species-target)),
2. `coords` is a 2D, n x 3 array of atomic coordinates
3. `edge_index0`, `edge_index1`, `edge_index2`... are the staged edge graphs described above
4. `contributions` is a vector of 1s and 0s, where 0 indicates the contributing atoms and 1 indicates the non-contributing atoms.

:::{tip}
The `contributions` vector here follows notation, opposite to that in generic models, please
see [this](#contributing-target) for more explanation.
:::

## Dealing with unmodifiable PBC and lattice vectors

For some reason if you cannot modify the GNN model to the format suggested above, you
can always use the wrapper-model approach defined in [generic model section](#generic-target).
To do so, you can write a wrapper model that takes in KIM API compliant arguments, and
uses layers explicitly to compute the forward pass. The edge graphs and vectors can be calculated
by using the unwrapped aperiodic coordinates from the KIM API, and setting the box dims and
lattice vectors to very large values, effectively making the system non-periodic.

A minimal example of how to manipulate the TorchScript layers, can be found [here](#appendix-torchscript-target).

## Nequip to KIM API

We provide a tool to convert NequIP models to KIM API compatible models. The tool is available
in the `kim_nequip` package. It can be used with a deployed NequIP model, along with the
configuration YAML file used to train tme model, to yield a functioning KIM API model.
The generated model can be used out of the box with LAMMPS, ASE and all other supported
simulators.

Example usage of the tool is as follows,

```shell
 kim-nequip-port  example.yaml --deployed-model deployed.pth --save-kim-model --verbose
```

where,

- `example.yaml` is the configuration file used to train the model
- `deployed.pth` is the deployed NequIP model you get from `nequip-deploy` command
- `--save-kim-model` is the flag to save the KIM API compatible model
- `--verbose` is the flag to print the progress of the conversion

Check [here](#nequip-port-target) for more details on the `kim-nequip-port` tool. The tool can be obtained
[here](https://github.com/ipcamit/kim_nequip).
