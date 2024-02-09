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
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    config["deployed-model"] = args.deployed_model
    config["out-name"] = args.out_name
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
    return



if __name__ == "__main__":
    main(running_as_script=True)
