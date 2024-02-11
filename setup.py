from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "kim_nequip/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="kim_nequip",
    version=version,
    description="Stripped down version of nequip 0.5.3 for KIM",
    author="Amit Gupta",
    python_requires=">=3.7",
    packages=find_packages(include=["kim_nequip", "kim_nequip.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "kim-nequip-port = kim_nequip.scripts.convert:main",
        ]
    },
    install_requires=[
        "numpy",
        "ase",
        "tqdm",
        "torch>=1.10.0,<1.13,!=1.9.0",
        "e3nn>=0.4.4,<0.6.0",
        "pyyaml",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        'contextvars;python_version<"3.7"',  # backport of contextvars for savenload
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "torch-runstats>=0.2.0",
        "torch-ema>=0.3.0",
    ],
    zip_safe=True,
)
