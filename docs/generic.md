(generic-target)=
# Generic Models
TorchML model driver gives the option to use completely generic models, by passing the raw 
inputs from the simulator to the model. The idea behind this model interface is to provide
a flexible option to the users to design models which cannot currently be implemented using
TorchML GNN or descriptor based scheme.

Given below is an example of Stillinger-Weber potential for implemented as Pytorch model,
```python
class StillingerWeberLayer(nn.Module):
    """
    Stillinger-Weber single species layer for Si atom for use in PyTorch model
    Before optimization, the parameter to be optimized need to be set using
    set_optim function. Forward method returns energy of the configuration
     and force array.
    """

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(15.2848479197914, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(0.6022245584, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.q = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(2.0951, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(2.51412, dtype=torch.float32))
        self.cutoff = nn.Parameter(torch.tensor(3.77118, dtype=torch.float32))
        self.lam = nn.Parameter(torch.tensor(45.5322, dtype=torch.float32))
        self.cos_beta0 = nn.Parameter(
            torch.tensor(-0.333333333333333, dtype=torch.float32)
        )

    def forward( self,
        species: torch.Tensor,
        coords: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_list: torch.Tensor,
        particle_contributing: torch.Tensor,
    ):
        total_conf_energy = energy( particle_contributing, coords, num_neighbors, neighbor_list,
            self.A, self.B, self.p, self.q, self.sigma, self.gamma, self.cutoff, self.lam, 
                                    self.cos_beta0,)
        forces = torch.autograd.grad([energy], [coords], create_graph=True)[0]
        if forces is None:
            forces = torch.tensor(-1)        
        return total_conf_energy, forces

```

The energy functions can be implemented as given in the [appendix](#sw-target). Let us go 
through the model inputs for now,

(species-target)=
## Species
The species tensor is a 1D vector containing the species of the atoms in the configuration.
Usually the species are represented as an index of values  0 to n - 1, where n is the number of 
species in the model. For example, in a system with two species, sae Si and O, the species
vector would look like,
```
[1, 1, 0, 0, 1, 1]
```
where, the first two atoms are of species O, and the next four are of species Si. Note that
indexing always assign the first element as zero, i.e. if you parameter file has species as
Si and O then Si would be assigned `0` and O would be assigned `1`, but if the parameter file
has species as O and S then O would be assigned `0` and Si would be assigned `1`. So 
be careful while defining the order of species in the parameter file.

:::{tip}
TorchML model driver also supports species as atomic numbers (i.e. Si as 14 and O as 8).
You can enable this by setting environment variable `KIM_MODEL_ELEMENTS_MAP` to `yes` at runtime.
:::

## Coordinates
The coordinates tensor is a 1D vector containing the coordinates of the atoms in the configuration.
This includes both contributing and non-contributing atoms. For n particles, the coordinates 
is a 3n length vector, with x, y, z coordinates of each atom. For example, in a system with
two atoms, the coordinates vector would look like,
```
[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
```
where, the first atom is at (0, 0, 0) and the second atom is at (1, 1, 1).

## Neighbour list and number of neighbors
In above example `num_neighbors`, is a 1D vector containing number of neighbors for all
contributing atoms. `neighbor_list` is a 1D vector containing the indices of the neighbors
for all contributing atoms.
For example, in a system with two contributing atoms, and three neighbors each, the `num_neighbors`
and `neighbor_list` would look like,
```python
num_neighbors = torch.tensor([3, 3], dtype=torch.int32)
neighbor_list = torch.tensor([1, 2, 3, 0, 2, 3], dtype=torch.int32)
```
where, particle 0 has neighbors with index 1, 2, 3 and particle 1 has neighbors with index 
0, 2, 3. On Python side these neighbor lists and number of neighbors can be obtained using
KLIFF's neighbor list utility as,
```python
from kliff.neighbor import NeighborList
nl = NeighborList(configuration, 3.77)
num_neighbors, neighbor_list = nl.get_numneigh_and_neighlist_1D()
```
where `configuration` is the `kliff.dataset.Configuration ` object. Please see KLIFF documentation 
for more details.

(contributing-target)=
## Contributing atoms
The `particle_contributing` vector is a 1D vector containing the information about whether the 
atom is contributing to the energy or not. 

:::{danger} 
Please note that the `particle_contributing` vector is modeled as a boolean vector, i.e. 1 for
contributing atoms and 0 for non-contributing atoms. This is opposite for GNN models, where
1 is for non-contributing atoms and 0 is for contributing atoms. This is because in GNNs 
the contribution/non-contribution is derived to benefit the "batching" in ML models 
(i.e. all contributing atoms batch to 0, non-contributing batch to 1).
:::

