"""
File: main.py
Author: Mughilan Muthupari
Creation Date: 2022-07-05

This file is the main file (no, seriously). It will train our models
given different Hydra configurations. Thanks to Hydra, it will also
make sure to save each configuration for traceability.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn

import hydra

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

