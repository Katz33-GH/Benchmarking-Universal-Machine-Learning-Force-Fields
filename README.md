# Benchmarking-Universal-Machine-Learning-Force-Fields

This repository benchmarks the performance of machine learning interatomic potentials—including MACE-MP-0, GRACE-1L-OAM, MatterSim v1 5M, and SevenNet-0—in predicting crystal structures across elemental, binary, and ternary systems. Evaluation is based on structural match quality using five metrics (Q1–Q5) computed via symmetry-aware structure matching.

## Q-Metric Definitions

| Metric | Description |
|--------|-------------|
| Q1 | True ground state structure appears *anywhere* in predicted results |
| Q2 | True ground state appears in the *three lowest-energy* predictions |
| Q3 | *Lowest-energy structure* predicted matches the true ground state |
| Q4 | All three reference structures appear in three lowest-energy structures (any order) |
| Q5 | Reference structures match three lowest-energy predicted structures in *exact relative energy order* |


## Requirements
Install the following dependencies before running the benchmark:

import torch
import yaml
from airsspy import SeedAtoms, Buildcell
from ipypb import ipb as tqdm
import json
from ase.io import write
from ase.optimize import BFGSLineSearch, sciopt, FIRE, BFGS, precon
from ase.constraints import UnitCellFilter, ExpCellFilter
from ase.filters import FrechetCellFilter
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import os
import csv
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

## Different calculators can be applied:

# MACE: 

from mace.calculators import mace_mp
calc = mace_mp(default_dtype="float64", device=device)


# GRACE-1L-OAM:

from tensorpotential.calculator import TPCalculator
calc = TPCalculator("/home/kzhong/.cache/grace/GRACE-1L-OAM_2Feb25")


# SevenNet-0 

from pymatgen.io.ase import AseAtomsAdaptor
adaptor = AseAtomsAdaptor()
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-0', device=device)


# MatterSim 

from pymatgen.io.ase import AseAtomsAdaptor
adaptor = AseAtomsAdaptor()
from mattersim.forcefield import MatterSimCalculator
calc=MatterSimCalculator(load_path="/home/kzhong/Benchmark/mattersim-v1.0.0-5M.pth", device=device)


