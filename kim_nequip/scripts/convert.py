""" Train a network."""
import logging
import argparse
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch

from kim_nequip.model import model_from_config
from kim_nequip.utils import Config
from kim_nequip.utils import load_file
from kim_nequip.utils._global_options import _set_global_options

from e3nn.util.jit import script

import shutil, os

default_config = dict(
    root="./",
    run_name="NequIP",
    wandb=False,
    wandb_project="NequIP",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        # "ForceOutput",
        # "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)


def main(args=None, running_as_script: bool = True):
    config = parse_command_line(args)
    trainer = fresh_start(config)
    print("Model Constructed. Exiting...")
    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a NequIP model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "--deployed-model",
        help="nequip deployed model you want to convert to KIM-API standard",
        type=str,
        default="deployed.pth"
    )
    parser.add_argument(
        "--out-name",
        help="Name of exported model that you want. Default is KIM_NequIP_model.pt",
        type=str,
        default="KIM_NequIP_model.pt"
    )
    parser.add_argument(
        "--save-kim-model",
        help="Give out a valid kim model",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--verbose",
        help="Logging verbosity level",
        action="store_true",
        default=False
    )

    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    config["deployed-model"] = args.deployed_model
    config["out-name"] = args.out_name
    config["give-kim-model"] = True
    config["verbose-conv"] = args.verbose
    return config


def fresh_start(config):
    # we use add_to_config cause it's a fresh start and need to record it
    # check_code_version(config, add_to_config=True)
    _set_global_options(config)

    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True
    )
    # save text file with same name as out-name
    out_name_dotted_list = config["out-name"].split(".")
    out_name_dotted_list[-1] = "txt"
    out_file_name = ".".join(out_name_dotted_list)
    with open(out_file_name, "w") as f:
        f.write("This is a KIM-API NequIP model for TorchML Model driver, below is the architecture:\n")
        f.write(str(final_model))
    # save model
    scritped_model = script(final_model)
    scritped_model.save(config["out-name"])
    print(f"Saved the model as {config['out-name']}")
    print(f"You can check the architecture in : {out_file_name}")
    try:
        deployed_model = torch.jit.load(config["deployed-model"])
    except:
        raise FileNotFoundError(f"Deployed model not found at {config['deployed-model']}")
    final_model = copy_weights(deployed_model, final_model, config)
    final_model.save(config["out-name"])
    if config["give-kim-model"]:
        save_kim_model(final_model, config)
    return

def copy_weights(deployed_model, final_model, config):
    deployed_model_params = (sum(p.numel() for p in deployed_model.parameters() if p.requires_grad))
    final_model_params = (sum(p.numel() for p in final_model.parameters() if p.requires_grad))
    assert deployed_model_params == final_model_params, \
        f"Number of parameters in deployed model: {deployed_model_params} and final model: {final_model_params} are not equal"
    deployed_state_dict = deployed_model.state_dict()
    final_state_dict = final_model.state_dict()
    deployed_key_list = list(deployed_state_dict.keys())
    for t in final_state_dict:
        for s in deployed_key_list:
            last_key = t #".".join(t.split(".")[1:])
            if (last_key in s) and len(last_key) > 0:
                if config["verbose-conv"]:
                    print(f"Copying {s} to {t}")
                final_state_dict[t] = deployed_state_dict[s]
            else:
                pass
    final_model.load_state_dict(final_state_dict)

    final_conv_indices = []
    i = 0
    for module in list(final_model.modules()):
        try:
            idx = module.conv.avg_num_neighbors
            if configs["verbose-conv"]:
                print(f"New model conv layer #: {idx}")
            final_conv_indices.append(i)
        except:
            pass
        i+=1

    deployed_conv_indices = []
    i = 0
    for module in list(deployed_model.modules()):
        try:
            idx = module.conv.avg_num_neighbors
            if configs["verbose-conv"]:
                print(f"Deployed model conv layer #: {idx}")
            deployed_conv_indices.append(i)
        except:
            pass
        i+=1

    # copy avg_num_neighbors
    for i in range(len(final_conv_indices)):
        list(final_model.modules())[final_conv_indices[i]].conv.avg_num_neighbors = (
            list(deployed_model.modules())[deployed_conv_indices[i]].conv.avg_num_neighbors)

    final_model.double()

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self,x,pos,edge_graph0, edge_graph1, edge_graph2, contributions):
            energy, forces = self.model(x,pos,edge_graph0,edge_graph1,edge_graph2,contributions)
            return energy, forces


    model_wrapped = WrappedModel(final_model)
    model_wrapped = torch.jit.script(model_wrapped)
    return model_wrapped

def save_kim_model(model, config):
    n_layers = config["num_layers"]
    cutoff = config["r_max"]
    n_species = config["num_types"]
    species = config["type_names"]
    model_name = config["out-name"]
    KIM_model_name = model_name.split(".")[0] + "__MO_000000000000_000"
    os.mkdir(KIM_model_name)
    shutil.copy(model_name, KIM_model_name)
    with open(f"{KIM_model_name}/file.param", "w") as f:
        f.write("# Num of elements\n")
        f.write(f"{n_species}\n")
        f.write(" ".join(species) + "\n\n")
        f.write("# Preprocessing Number of Conv layers\n")
        f.write("Graph\n\n")
        f.write("# Cutoff Distance (Infl. dist = cutoff x n_conv_layers) and n_conv_layer\n")
        f.write(f"{cutoff}\n{n_layers}\n\n")
        f.write("# Model name\n")
        f.write(f"{model_name}\n\n")
        f.write("# Return Forces\n")
        f.write("True\n\n")
        f.write("# Number of Inputs (n_conv_layers + other inputs)\n# 3 conv layers  + 1 element embedding\n")
        f.write(f"{n_layers + 3}\n\n")
        f.write("# Descriptors\nNone\n")

    with open(f"{KIM_model_name}/CMakeLists.txt", "w") as f:
        f.write("cmake_minimum_required(VERSION 3.10)\n")
        f.write("list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})\n")
        f.write("find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)\n")
        f.write("kim_api_items_setup_before_project(ITEM_TYPE \"portableModel\")\n")
        f.write(f"project({KIM_model_name})\n")
        f.write("kim_api_items_setup_after_project(ITEM_TYPE \"portableModel\")\n")
        f.write("add_kim_api_model_library(\n")
        f.write("  NAME            ${PROJECT_NAME}\n")
        f.write("  DRIVER_NAME     \"TorchML__MD_173118614730_000\"\n")
        f.write("  PARAMETER_FILES \"file.param\"\n")
        f.write(f"                  \"{model_name}\")\n")



if __name__ == "__main__":
    main(running_as_script=True)
