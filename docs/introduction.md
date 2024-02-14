# TorchML Model Driver for KIM API

With the new [TorchMl model driver](https://openkim.org/id/MD_173118614730_000) ML models are
at parity with conventional models in OpenKIM model repository. The model driver brings
the same ease of installation and use of ML models that conventional models enjoy. T
For example the NequIP Si model ` TorchML_NequIP_GuptaTadmorMartiniani_2024_Si__MO_196181738937_000`
can be installed as,

:::{tip}
kim-api-collections-management install user TorchML_NequIP_GuptaTadmorMartiniani_2024_Si\_\_MO_196181738937_000
:::

## Dependencies and Environment

This should install both the model and the model driver.
At present the model driver has the following dependencies,

1. KIM API (compulsory)
2. libtorch CXX API (v1.13, compulsory)
3. libtorchscatter and libtorchsparse (not the python packages, but the CXX API, compulsory for GNN)
4. libdescriptor (CXX API, compulsory for descriptor based models)

Out of these, only KIM-API and libtorch are must for all models. The other two are model specific.
You can install the required environment using the script provided in the model driver repository ([linked here](https://openkim.org/files/MD_173118614730_000/install_dependencies.sh)).

Optional environment variables for customizing the installation are,

| Variable                  | Description                                                                                                                                                                                                                                                                                                         |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `KIM_MODEL_MPI_AWARE`     | if it is set to `yes`, then the model driver would need MPI library at installation time, and it would install model driver with capabilities to assign multiple GPUs as per the MPI rank of the running process. In short each rank gets the `GPU# = num_gpu%rank_id` (Please see README on the model driver page) |
| `KIM_MODEL_DISABLE_GRAPH` | If set to `yes` it would compile the model driver wihtout requiring libtorchscatter and libtorchsparse libraries. This means that the model driver will not support the GNN                                                                                                                                         |

## Model locality and the need to port

### TorchScript

Pytorch provides a dynamic and easy to use interface for Ml models.
However, this comes at the cost of reproducibility as the model can have dynamic branching and global states that influence its output.
Therefor, to ensure reproducibility, the model has to be converted to a static graph, which is then used for inference.
This is done using the `torch.jit.script` module, which converts the model to a TorchScript model.
TorchScript is a serialized intermediate representation of a Pytorch model, which can be used directly in the C++ API of Pytorch (libtorch).

```python
import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = MyModel()
scripted_model = torch.jit.script(model)
# save model to file
scripted_model.save("model.pt")
print(scripted_model)
```

This should show the result of serialization as,

```text
RecursiveScriptModule(
  original_name=MyModel
  (fc1): RecursiveScriptModule(original_name=Linear)
  (fc2): RecursiveScriptModule(original_name=Linear)
)
```

TorchScript serialization fails if the model has dynamic behavior like,

```python
#...
    def forward(self, x, y):
        for i in range(y):
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#...
```

where the output depends on the value of `y`. As the value of `y` is not known at the time of serialization, the model cannot be serialized.

```text
RuntimeError:
all inputs of range must be ints, found Tensor (inferred) in argument 0:
  File "<ipython-input-3-47186d33535a>", line 9
    def forward(self, x,y):
        for i in range(y):
                 ~~~~~~~ <--- HERE
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
```

Therefore, to ensure portability and reproducibility, TorchML model driver only supports
the TorchScripted models.
In principle the TorchML model driver supports any and all TorchScript models. But
the restriction is on the inputs and outputs of the model. This is to keep a uniform
API for the simulators to interact with the model driver.

### Model locality

The biggest restriction the KIM API applies on the models is the locality of the model.
That is, the model should be able to compute the energy and the forces of the system
solely from positions, and species of atoms from within an "influence distance". The
model should not require any global information, for example the box size, or the lattice
vectors. Vast majority of ML models can be expressed as pure local models and hence this
does not pose a significant restriction, but it means that certain models need to be
reformulated to fully comply with the KIM API.

Biggest culprit is the periodic boundary conditions and lattice vectors, where the model
does not use purely coordinate information, but rather uses the lattice vectors to compute
the unwrapped distances of atoms. As KIM API does not give the lattice vectors, this
poses a restriction.

For graph neural networks, we present a novel solution to the problem by using
"staged graph convolution" (explained more in GNN section), which not only make the GNNs local, but also makes them
embarrassingly parallel. We provide `kim_nequip` library to convert NequIP models to
KIM API compatible models, for other popular structures, like MACE, similar tools will
be released soon.
