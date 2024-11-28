import botorch.test_functions, math
import torch
from botorch.utils.transforms import unnormalize
from typing import List, Optional, Tuple, Union
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
import math
from typing import Optional
from scipy.stats import norm
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
import scipy.integrate as integrate
import scipy.special as special
import numpy as np
from torchquad import MonteCarlo, set_up_backend, Trapezoid, Simpson
import sys
import os
from typing import Optional
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
from botorch.exceptions.warnings import InputDataWarning
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann, Ackley
import time
import warnings
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from conformalbo.acquisitions import qConformalExpectedImprovement
from conformalbo.utils import construct_acq_fn, construct_acq_fn_mod
import hydra
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from botorch.exceptions.warnings import InputDataWarning
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf

from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
import argparse


