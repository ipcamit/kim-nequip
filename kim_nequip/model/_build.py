import inspect
from typing import Optional

from kim_nequip.data import AtomicDataset
from kim_nequip.data.transforms import TypeMapper
from kim_nequip.nn import GraphModuleMixin
from kim_nequip.utils import load_callable, instantiate


def model_from_config(
    config,
    initialize: bool = False,
    # dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> GraphModuleMixin:
    """Build a model based on `config`.

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``dataset``: if ``initialize`` is True, the dataset
     - ``deploy``: whether the model object is for deployment / inference

    Args:
        config
        initialize (bool): whether ``model_builders`` should be instructed to initialize the model
        dataset: dataset for initializers if ``initialize`` is True.
        deploy (bool): whether ``model_builders`` should be told the model is for deployment / inference

    Returns:
        The build model.
    """
    # Pre-process config
    type_mapper = None
    try:
        type_mapper, _ = instantiate(TypeMapper, all_args=config)
    except RuntimeError:
        pass

    if type_mapper is not None:
        if "num_types" in config:
            assert (
                config["num_types"] == type_mapper.num_types
            ), "inconsistant config & dataset"
        if "type_names" in config:
            assert (
                config["type_names"] == type_mapper.type_names
            ), "inconsistant config & dataset"
        config["num_types"] = type_mapper.num_types
        config["type_names"] = type_mapper.type_names

    # Build
    # print(f"Model Builder Config: {config} {config.get('model_builders')}")
    builders = [
        load_callable(b, prefix="kim_nequip.model")
        for b in config.get("model_builders", [])
    ]

    # remove ForceOutput and RescaleEnergyEtc
    for b in builders:
        if b.__name__ == "ForceOutput" or b.__name__ == "RescaleEnergyEtc":
            builders.remove(b)

    model = None

    for builder in builders:
        print(f"Building model with {builder}")
        pnames = inspect.signature(builder).parameters
        params = {}
        if "initialize" in pnames:
            params["initialize"] = initialize
        if "config" in pnames:
            params["config"] = config
        if "model" in pnames:
            if model is None:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model"
                )
            params["model"] = model
        else:
            if model is not None:
                raise RuntimeError(
                    f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't"
                )
        model = builder(**params)
    return model
