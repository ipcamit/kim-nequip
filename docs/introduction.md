# TorchML Model Driver for KIM API
With the new [TorchMl model driver](https://openkim.org/id/MD_173118614730_000) ML models are
at parity with conventional models in OpenKIM model repository. The model driver brings
the same ease of installation and use of ML models that conventional models enjoy. T
For example the NequIP Si model ` TorchML_NequIP_GuptaTadmorMartiniani_2024_Si__MO_196181738937_000`
can be installed as,

```{bash}
kim-api-collections-management install user TorchML_NequIP_GuptaTadmorMartiniani_2024_Si__MO_196181738937_000
```

This should install both the model and the model driver. 
At present the model driver has the following dependencies,
1. KIM API (compulsory)
2. libtorch CXX API (v1.13, compulsory)
3. libtorchscatter and libtorchsparse (not the python packages, but the CXX API, compulsory for GNN)
4. libdescriptor (CXX API, compulsory for descriptor based models)

Out of these, only KIM-API and libtorch are the a must for all models. The other two are model specific.
You can install the required environment using the script provided in the model driver repository ([linked here](https://openkim.org/files/MD_173118614730_000/install_dependencies.sh)).

Optional environment variables for customizing the installation are,

| Variable | Description                                                                                                                                                                                                                                                                                                         |
|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`KIM_MODEL_MPI_AWARE`| if it is set to `yes`, then the model driver would need MPI library at installation time, and it would install model driver with capabilities to assign multiple GPUs as per the MPI rank of the running process. In short each rank gets the `GPU# = num_gpu%rank_id` (Please see README on the model driver page) |
|`KIM_MODEL_DISABLE_GRAPH`| If set to `yes` it would compile the model driver wihtout requiring libtorchscatter and libtorchsparse libraries. This means that the model driver will not support the GNN|



