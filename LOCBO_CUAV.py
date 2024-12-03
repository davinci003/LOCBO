import sys
import argparse
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
import numpy as np
import random
from scipy.spatial.distance import cdist
import os
import torch
from botorch.optim import optimize_acqf
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
import scipy.integrate as integrate
import scipy.special as special

parser = argparse.ArgumentParser(description="Process simulation parameters.")
parser.add_argument("num_user", type=int, help="Number of users")
parser.add_argument("thermal_noise_arg", type=float, help="Thermal noise argument")
parser.add_argument("max_tx_power_arg", type=int, help="Maximum transmission power argument")
parser.add_argument("number_channels", type=int, help="Number of channels")
parser.add_argument("BS_UAV_ratio", type=float, help="Base station to UAV ratio")
parser.add_argument("cuda_device", type=int, help="CUDA device ID")
parser.add_argument("switch_fit_model", type=int, help="Switch fit model (0 or 1)")
args = parser.parse_args()
num_user = args.num_user
thermal_noise_arg = args.thermal_noise_arg
max_tx_power_arg = args.max_tx_power_arg
number_channels = args.number_channels
BS_UAV_ratio = args.BS_UAV_ratio
cuda_device = args.cuda_device
switch_fit_model = args.switch_fit_model

def hexagonal_layout_two_layers():
    coordinates = []
    offset = 1
    coordinates.append((0 + offset * math.cos(math.pi / 2), 0 + offset * math.sin(math.pi / 2)))
    coordinates.append((0 + offset * math.cos(math.pi * 7 / 6), 0 + offset * math.sin(math.pi * 7 / 6)))
    coordinates.append((0 + offset * math.cos(math.pi * 11 / 6), 0 + offset * math.sin(math.pi * 11 / 6)))

    coordinates.append((0 + offset * math.cos(math.pi / 2), 200 + offset * math.sin(math.pi / 2)))
    coordinates.append((0 + offset * math.cos(math.pi * 7 / 6), 200 + offset * math.sin(math.pi * 7 / 6)))
    coordinates.append((0 + offset * math.cos(math.pi * 11 / 6), 200 + offset * math.sin(math.pi * 11 / 6)))

    coordinates.append((171 + offset * math.cos(math.pi / 2), 100 + offset * math.sin(math.pi / 2)))
    coordinates.append((171 + offset * math.cos(math.pi * 7 / 6), 100 + offset * math.sin(math.pi * 7 / 6)))
    coordinates.append((171 + offset * math.cos(math.pi * 11 / 6), 100 + offset * math.sin(math.pi * 11 / 6)))
    return coordinates

BSs = hexagonal_layout_two_layers()

def deploy_users_in_donut(BS_orient, num_users, R_inner, R_outer, offset, idx):
    np.random.seed(idx)
    if BS_orient % 3 == 0:
        theta = np.random.uniform(1 / 6 * np.pi, 5 / 6 * np.pi, num_users)
    elif BS_orient % 3 == 1:
        theta = np.random.uniform(5 / 6 * np.pi, 10 / 6 * np.pi, num_users)
    else:
        theta = np.random.uniform(10 / 6 * np.pi, 13 / 6 * np.pi, num_users)
    u = np.random.uniform(0, 1, num_users)
    r = np.sqrt(u * (R_outer ** 2 - R_inner ** 2) + R_inner ** 2)
    x = r * np.cos(theta) + offset[0]
    y = r * np.sin(theta) + offset[1]
    return x, y


def generate_random_points(num_points, box_size, BSs):
    points = []
    for i in range(len(BSs)):
        for j in range(int(num_points / len(BSs))):
            x, y = deploy_users_in_donut(i, 1, 10, box_size[0], BSs[i], i * len(BSs) + j)
            points.append((x[0], y[0]))
    return points

num_UE_per_BS = num_user
num_points = len(BSs) * num_UE_per_BS
box_size = [120, 120]

# Generate random points
UEs = generate_random_points(num_points, box_size, BSs)
num_points = len(UEs)

def UAV(num_per_corridor=7):
    corridor = {
        0: [[-110, -100], [-100, 300]],
        1: [[-100, 270], [300, 310]],
        2: [[270, 280], [-100, 300]],
        3: [[-100, 270], [-110, -100]]
    }
    UAV = []

    for i in range(4):
        x_start, x_end = corridor[i][0]
        y_start, y_end = corridor[i][1]
        x_points = np.linspace(x_start, x_end, num_per_corridor)
        y_points = np.linspace(y_start, y_end, num_per_corridor)
        for x, y in zip(x_points, y_points):
            UAV.append((x, y))
    return UAV

UAVs = UAV()

def small_fading(BS, UE, UAV, idx):  # return dBm
    np.random.seed(10*idx)
    BS = np.array(BS)
    UE = np.array(UE)
    UAV = np.array(UAV)
    r_Small_fading_array = 1 / math.sqrt(2) * np.random.normal(0, 1,
                                                               (len(BS), len(UAV) + len(UE)))  # mu, standard deviation
    i_Small_fading_array = 1 / math.sqrt(2) * np.random.normal(0, 1, (len(BS), len(UAV) + len(UE)))
    Small_fading_array = 10 * np.log10(
        (r_Small_fading_array * r_Small_fading_array + i_Small_fading_array * i_Small_fading_array))
    Small_fading_array[:, len(UE):] = 0
    return Small_fading_array


def xy_dist(BS, UE, UAV):
    BS = np.array(BS)
    UE = np.array(UE)
    UAV = np.array(UAV)
    Total_UE = np.concatenate((UE, UAV), axis=0)
    distances = cdist(BS, Total_UE)
    return distances

def xyz_dist(BS_height, UE_height, UAV_height, BS, UE, UAV):
    BS = np.array(BS)
    UE = np.array(UE)
    UAV = np.array(UAV)
    Total_UE = np.concatenate((UE, UAV), axis=0)
    distances = cdist(BS, Total_UE)
    Arr_UE_height = (BS_height - UE_height) * np.ones((len(BS), len(UE)))
    Arr_UAV_height = (BS_height - UAV_height) * np.ones((len(BS), len(UAV)))
    Arr_tot_height = np.concatenate((Arr_UE_height, Arr_UAV_height), axis=1)
    xyz_distance = np.sqrt((Arr_tot_height * Arr_tot_height) + (distances * distances))
    return xyz_distance

def large_fading(BS_height, UE_height, UAV_height, BS, UE, UAV):  # return dBm
    BS = np.array(BS)
    UE = np.array(UE)
    UAV = np.array(UAV)
    Total_UE = np.concatenate((UE, UAV), axis=0)
    distances = cdist(BS, Total_UE)
    Arr_UE_height = (BS_height - UE_height) * np.ones((len(BS), len(UE)))
    Arr_UAV_height = (BS_height - UAV_height) * np.ones((len(BS), len(UAV)))
    Arr_tot_height = np.concatenate((Arr_UE_height, Arr_UAV_height), axis=1)
    xyz_distance = np.sqrt((Arr_tot_height * Arr_tot_height) + (distances * distances))
    large_fading_UE = 13.54 + 39.08 * np.log10(xyz_distance[:, :len(UE)]) + 20 * np.log10(2)
    large_fading_UAV = 28 + 22 * np.log10(xyz_distance[:, len(UE):]) + 20 * np.log10(2)
    large_fading_result = np.concatenate((large_fading_UE, large_fading_UAV), axis=1)
    return large_fading_result

def vert_angle(xy_dist, BS_height, UE_height, UAV_height):
    UE_verticle_angle = 180 / math.pi * np.arctan((UE_height - BS_height) / xy_dist[:, :num_points])
    UAV_verticle_angle = 180 / math.pi * np.arctan((UAV_height - BS_height) / xy_dist[:, num_points:])
    vert_angle_result = np.concatenate((UE_verticle_angle, UAV_verticle_angle), axis=1)
    return vert_angle_result

def angle_between_vectors(vector1, vector2):  # return 0~180 degree
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def calculate_vectors(group_a, group_b):
    vectors = []
    for point_a in group_a:
        for point_b in group_b:
            vector = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
            vectors.append(vector)
    vectors = np.array(vectors)
    reshaped_array = np.reshape(vectors, (len(group_a), len(group_b), 2))
    return reshaped_array

def horz_angle(BSs, UEs, UAVs):
    Tot_UE = np.concatenate((UEs, UAVs), axis=0)
    vector_btw_BS_UE = calculate_vectors(BSs, Tot_UE)
    basis = np.array(
        [[0, 1], [np.cos(7 / 6 * np.pi), np.sin(7 / 6 * np.pi)], [np.cos(11 / 6 * np.pi), np.sin(11 / 6 * np.pi)]])
    vector_BS = np.tile(basis, (int(len(BSs) / 3), len(Tot_UE)))
    vector_BS = np.reshape(vector_BS, (len(BSs), len(Tot_UE), 2))
    horizon_angle = np.zeros((len(BSs), len(Tot_UE)))
    for i in range(len(BSs)):
        for j in range(len(Tot_UE)):
            horizon_angle[i, j] = angle_between_vectors(vector_btw_BS_UE[i, j], vector_BS[i, j])
    return horizon_angle

def antenna_gain(vert_angle, horz_angle, allocated_vert_angle):  # return dBm
    A_max = 8
    vert_theta_3dB = 10
    horz_theta_3dB = 65
    allocated_vert_angle = np.repeat(allocated_vert_angle, vert_angle.shape[1], axis=1)
    gain = A_max - 12 / (vert_theta_3dB ** 2) * np.multiply((allocated_vert_angle - vert_angle),
                                                            (allocated_vert_angle - vert_angle)) - 12 / (
                       horz_theta_3dB ** 2) * np.multiply(horz_angle, horz_angle)
    gain = np.clip(gain, -30, 8)
    return gain

def calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, fading_l, Tx_power):
    result_sinr = RSS + fading_s
    return result_sinr

def calculate_SINR_array(RSS, RSS_with_s_fading, thermal_noise):
    SINR = np.zeros(RSS.shape[1])
    RSS_with_s_fading = np.power(10, RSS_with_s_fading / 10)
    for i in range(RSS.shape[1]):
        max_idx = np.argmax(RSS[:, i])
        P_signal = RSS_with_s_fading[max_idx, i]
        Interference = np.sum(RSS_with_s_fading[:, i]) - P_signal
        SINR[i] = P_signal / ((Interference) + thermal_noise)
    SINR = np.log2(1 + SINR)
    return SINR

def calculate_SINR(RSS, RSS_with_s_fading, thermal_noise):
    SINR = np.zeros(RSS.shape[1])
    RSS_with_s_fading = np.power(10, RSS_with_s_fading / 10)
    for i in range(RSS.shape[1]):
        max_idx = np.argmax(RSS[:, i])
        P_signal = RSS_with_s_fading[max_idx, i]
        Interference = np.sum(RSS_with_s_fading[:, i]) - P_signal
        SINR[i] = P_signal / (Interference + thermal_noise)
    SINR = np.log2(1 + SINR)
    GUE_lambda = BS_UAV_ratio
    avg_sinr = (GUE_lambda) * np.sum(SINR[:num_points]) / (num_points) + (1 - GUE_lambda) * np.sum(
        SINR[num_points:]) / (200)
    return avg_sinr

class objective:
    def __init__(self, BSs, UEs, UAVs, thermal_noise, fading_l, angle_vertical, fixed_horz_angle):
        self.BS_height = 25
        self.UE_height = 1.5
        self.UAV_height = 150
        self.BSs = BSs
        self.UEs = UEs
        self.UAVs = UAVs
        self.thermal_noise = thermal_noise
        self.angle_vertical = angle_vertical
        self.fixed_horz_angle = fixed_horz_angle
        self.dist_xy = xy_dist(self.BSs, self.UEs, self.UAVs)  # size 57*1055
        self.fading_l = large_fading(self.BS_height, self.UE_height, self.UAV_height, self.BSs, self.UEs, self.UAVs)
        self.sinr_array = 0
        self.RSS_with_s_fading = 0

    def observation(self, X, fading_idx):
        fading_s = small_fading(self.BSs, self.UEs, self.UAVs, fading_idx)
        Tx_power = X[0:int(X.shape[0] / 2)] * max_tx_power_arg + 6.0
        Tilting_ang = X[int(X.shape[0] / 2):] * 180.0 - 90.0
        ant_gain = antenna_gain(self.angle_vertical, self.fixed_horz_angle, Tilting_ang)
        RSS = np.repeat(Tx_power, self.dist_xy.shape[1], axis=1) + ant_gain - self.fading_l
        RSS_with_s_fading = calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, self.fading_l, Tx_power)
        avg_sinr = calculate_SINR(RSS, RSS_with_s_fading, self.thermal_noise)
        return avg_sinr

    def observation_w_intial_channels(self, X, start, end):
        sum_n_sinr = []
        Tx_power = X[0:int(X.shape[0] / 2)] * max_tx_power_arg + 6.0
        Tilting_ang = X[int(X.shape[0] / 2):] * 180.0 - 90.0
        ant_gain = antenna_gain(self.angle_vertical, self.fixed_horz_angle, Tilting_ang)
        fading_idx = np.random.randint(0, 1000)
        fading_s = small_fading(self.BSs, self.UEs, self.UAVs, fading_idx)
        RSS = np.repeat(Tx_power, self.dist_xy.shape[1], axis=1) + ant_gain - self.fading_l
        RSS_with_s_fading = calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, self.fading_l, Tx_power)
        avg_sinr = calculate_SINR(RSS, RSS_with_s_fading, self.thermal_noise)
        sum_n_sinr.append(avg_sinr)
        return np.array(sum_n_sinr).mean()

    def observation_w_N_channels(self, X, start, end):
        sum_n_sinr = []
        Tx_power = X[0:int(X.shape[0] / 2)] * max_tx_power_arg + 6.0
        Tilting_ang = X[int(X.shape[0] / 2):] * 180.0 - 90.0
        ant_gain = antenna_gain(self.angle_vertical, self.fixed_horz_angle, Tilting_ang)

        for i in range(start, end):
            fading_idx = i
            fading_s = small_fading(self.BSs, self.UEs, self.UAVs, fading_idx)
            RSS = np.repeat(Tx_power, self.dist_xy.shape[1], axis=1) + ant_gain - self.fading_l
            RSS_with_s_fading = calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, self.fading_l, Tx_power)
            avg_sinr = calculate_SINR(RSS, RSS_with_s_fading, self.thermal_noise)
            sum_n_sinr.append(avg_sinr)
            self.sinr_array = calculate_SINR_array(RSS, RSS_with_s_fading, self.thermal_noise)
            self.RSS_with_s_fading = RSS_with_s_fading
        return np.array(sum_n_sinr).mean()

    def MC_true_fun(self, X):
        sum_n_sinr = []
        Tx_power = X[0:int(X.shape[0] / 2)] * max_tx_power_arg + 6.0
        Tilting_ang = X[int(X.shape[0] / 2):] * 180.0 - 90.0
        ant_gain = antenna_gain(self.angle_vertical, self.fixed_horz_angle, Tilting_ang)
        for i in range(100):
            fading_idx = i
            fading_s = small_fading(self.BSs, self.UEs, self.UAVs, fading_idx)
            RSS = np.repeat(Tx_power, fading_s.shape[1], axis=1) + ant_gain - self.fading_l
            RSS_with_s_fading = calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, self.fading_l, Tx_power)
            avg_sinr = calculate_SINR(RSS, RSS_with_s_fading, self.thermal_noise)
            sum_n_sinr.append(avg_sinr)
        if i == 999:
            print(avg_sinr)
        return np.array(sum_n_sinr).mean()

thermal_noise = 10**(-1*(thermal_noise_arg))
rand_idx = 2
fading_s = small_fading(BSs, UEs, UAVs, rand_idx) #size 57*1055
dist_xy = xy_dist(BSs, UEs, UAVs) #size 57*1055
tilting_angle = np.ones((dist_xy.shape[0],1))*0.0
Tx_power = np.ones((dist_xy.shape[0],1))*46
fading_l = large_fading(BS_height=25, UE_height=1.5, UAV_height=150,BS = BSs, UE = UEs, UAV = UAVs)
angle_vertical = vert_angle(xy_dist=dist_xy, BS_height=25, UE_height=1.5, UAV_height=150)
fixed_horz_angle = horz_angle(BSs, UEs, UAVs)
ant_gain = antenna_gain(angle_vertical, fixed_horz_angle, tilting_angle)
RSS = np.repeat( Tx_power , num_points + len(UAVs), axis=1) + ant_gain - fading_l
RSS_with_s_fading = calculate_RSS_with_s_fading(RSS, ant_gain, fading_s, fading_l, Tx_power)
SINR_arr=calculate_SINR_array(RSS, RSS_with_s_fading,thermal_noise)
result = calculate_SINR(RSS, RSS_with_s_fading,thermal_noise)
Tot_list = []
Tx_power = np.ones((dist_xy.shape[0],1))
tilting_angle = 0.5*np.ones((dist_xy.shape[0],1))
query_x = np.concatenate((Tx_power, tilting_angle), axis = 0)
query_x = np.array(query_x)
mc_array = []
num_observing_channels = 3
O_1 = objective( BSs, UEs, UAVs, thermal_noise, fading_l, angle_vertical, fixed_horz_angle)

device = torch.device("cuda:"+str(cuda_device) if torch.cuda.is_available() else "cpu")
dtype = torch.float32
SMOKE_TEST = os.environ.get("SMOKE_TEST")

def cuav_generate_initial_data(num_init_points, num_channels):
    train_x = torch.zeros(num_init_points, len(BSs)*2, device=device, dtype=dtype)
    train_obj = torch.zeros(num_init_points,1, device=device, dtype=dtype)
    O_1 = objective( BSs, UEs, UAVs, thermal_noise, fading_l, angle_vertical, fixed_horz_angle)
    for num_data in range(num_init_points):
      np.random.seed(num_data)
      inst_train_x = np.concatenate((np.random.rand(len(BSs),1), np.random.rand(len(BSs),1)), axis =0 )
      train_x[num_data,:] = torch.tensor(inst_train_x).squeeze()
      f_x = O_1.observation_w_intial_channels(inst_train_x, num_channels,num_channels*2)
      f_x = torch.tensor([f_x], device=device, dtype=dtype)
      train_obj[num_data] = f_x
    best_observed_value = train_obj.max().item()
    random.seed()
    return train_x, train_obj, best_observed_value# return [10,6], [10,1], [10,1]

def initialize_model(train_x, train_obj, state_dict=None):
    model_obj = SingleTaskGP(train_x, standardize(train_obj))
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj, standardize(train_obj).max()
    
def online_cbo_alpha(train_x, train_obj, alpha_desired, model_ei, iteration, eta_init, reg_param, length_scale, decay,
                     err_list, n_init):
    test_x = torch.unsqueeze(train_x[-1], 0)
    posterior = model_ei.posterior(test_x, observation_noise=True)
    post_mean, post_var = posterior.mean, posterior.variance
    eta_func_const = 2
    c_t = alpha_desired
    if iteration == 1:
        g_t_x = c_t
    else:
        t_values = np.arange(1, iteration)
        eta_list = eta_init / (t_values + 1) ** decay
        eta_func_list = eta_init / (t_values + 1) ** decay
        array1 = np.array(eta_list)
        array2 = np.array([desired_alpha - x for x in err_list])
        c_t = np.sum(array1 * array2) + c_t
        lambda_term_list = []
        kernel_ARC_list = []
        array1 = np.array(eta_func_list)
        array2 = np.array([desired_alpha - x for x in err_list])
        eta_term_list = array1 * array2
        cum_prod = 1
        for i in range(1, iteration):
            if i == 1:
                cum_prod = eta_list[-(i - 1)] ** 0
            else:
                cum_prod = (1 - reg_param * eta_list[-(i - 1)]) * cum_prod
            lambda_term_list.append(cum_prod)
            kernel_ARC_list.append(
                torch.exp(-torch.norm(train_x[n_init + i - 1] - test_x).to('cpu') ** 2 / (length_scale ** 2)))
        lambda_term_list = lambda_term_list[::-1]
        lambda_term_list = np.array(lambda_term_list)
        kernel_ARC_list = np.array(kernel_ARC_list)
        g_list = lambda_term_list * eta_term_list * kernel_ARC_list
        g_t_x = eta_func_const * kappa * np.sum(g_list) + c_t
    if g_t_x < 0:
        g_t_x = 0
    if g_t_x < 1:
        u = norm.ppf(1-g_t_x/2)
        l = norm.ppf(g_t_x/2)
    else:
        l = norm.ppf(1 - g_t_x / 2)
        u = norm.ppf(g_t_x / 2)
    u = u * torch.sqrt(post_var) + post_mean
    l = l * torch.sqrt(post_var) + post_mean
    if train_obj>=l and train_obj<=u:
        err_t = 0
    else:
        err_t = 1
    err_list.append(err_t)
    return err_list

bounds = torch.tensor([[0.0] * len(BSs) + [0.0] * len(BSs), [1.0] * len(BSs) +[1.0] * len(BSs) ], device=device, dtype=dtype)

BATCH_SIZE = 1 if not SMOKE_TEST else 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32

def integrand_middle(x, a, b, sigma, l, u, sigma_y, best_f, alpha, desired_alpha):
    return max(x-best_f, 0)*(1-desired_alpha)*(1/u-l)*1/(2*a)*(special.erf(1/(np.sqrt(2)*sigma)*(-x+a*u+b)) - special.erf(1/(np.sqrt(2)*sigma)*(-x+a*l+b)))

def integrand_tale(x, a, b, sigma, l, u, sigma_y, alpha, mu_y, best_f, desired_alpha):
    if torch.is_tensor(best_f):
        best_f = best_f.detach().cpu().numpy()
    val_pi = np.pi
    A_square = (a ** 2 / (2 * sigma ** 2)) ** 2
    B = (-b + x) / (a)
    C_square = (1 / (2 * sigma_y ** 2)) ** 2
    D = mu_y
    upper = 1/(2*val_pi*sigma*sigma_y)*0.5 * np.sqrt(val_pi / (A_square + C_square)) * np.exp(
        -A_square * C_square / (A_square + C_square) * (B - D) ** 2) * (1 -special.erf(
        (-B * A_square - D * C_square + (A_square + C_square) * u) / np.sqrt(A_square + C_square)))
    lower = 1/(2*val_pi*sigma*sigma_y)*0.5 * np.sqrt(val_pi / (A_square + C_square)) * np.exp(
        -A_square * C_square / (A_square + C_square) * (B - D) ** 2) * (1+special.erf(
        (-B * A_square - D * C_square + (A_square + C_square) * l) / np.sqrt(A_square + C_square)))
    return desired_alpha/alpha*(upper+lower)* max(x-best_f, 0)

def online_cbo_fixed_dim_get_observation(num_channels, best_x ,fixed_dim, num_drawn, device, model, train_x, train_obj, acq_input_best_value,
                                         desired_alpha, n_init, iteration, err_list, length_scale, eta_init, reg_param, kappa, decay):
    eta_init, reg_param, length_scale, kappa, decay = eta_init, reg_param, length_scale, kappa, decay
    device = device
    best_train_x = np.expand_dims(np.squeeze(best_x), axis =0)
    best_train_x = np.repeat(best_train_x, num_drawn, axis=0)
    np.random.seed()
    ei_list = torch.tensor([]).to(device)
    eta_list = []
    eta_func_list = []
    eta_func_const = 2
    power = np.random.rand(num_drawn)
    theta = np.random.rand(num_drawn)
    best_train_x[:, (fixed_dim - 1) % len(BSs)] = power
    best_train_x[:, (fixed_dim - 1) % len(BSs) + len(BSs)] = theta
    for t in range(1, iteration):
        eta_list.append(eta_init / (t + 1) ** (decay))
        eta_func_list.append(eta_init / (t + 1) ** (decay))
    repeat_iter = 1
    history_x = train_x
    history_y = standardize(train_obj)
    GP = model
    kernel = GP.covar_module

    for j in range(int(num_drawn/repeat_iter)):
        if j == 0:
            with torch.no_grad():
                c_t = desired_alpha
                test_x = torch.tensor(np.expand_dims(best_train_x[repeat_iter*j:repeat_iter*j+repeat_iter,:], axis=1)).to(device)
                if iteration == 1:
                    g_t_x = c_t
                    alpha_t_x = g_t_x
                else:
                    t_values = np.arange(1, iteration)
                    eta_list = eta_init / (t_values + 1) ** decay
                    eta_func_list = eta_init / (t_values + 1) ** decay
                    array1 = np.array(eta_list)
                    array2 = np.array([desired_alpha - x for x in err_list])
                    c_t = np.sum(array1 * array2) + c_t
                    lambda_term_list = []
                    kernel_ARC_list = []
                    array1 = np.array(eta_func_list)
                    array2 = np.array([desired_alpha - x for x in err_list])
                    eta_term_list = array1 * array2
                    for i in range(1, iteration):
                        if i == 1:
                            cum_prod = eta_list[-(i - 1)] ** 0
                        else:
                            cum_prod = (1 - reg_param * eta_list[-(i - 1)]) * cum_prod
                        lambda_term_list.append(cum_prod)
                        kernel_ARC_list.append(torch.exp(-torch.norm(history_x[n_init + i - 1] - test_x).to('cpu') ** 2 / (length_scale ** 2)))
                    lambda_term_list = lambda_term_list[::-1]
                    lambda_term_list = np.array(lambda_term_list)
                    kernel_ARC_list = np.array(kernel_ARC_list)
                    g_list = lambda_term_list * eta_term_list * kernel_ARC_list
                    g_t_x = eta_func_const * kappa * np.sum(g_list) + c_t
                    posterior = model.posterior(test_x, observation_noise=True)
                    post_mean, post_var = posterior.mean, posterior.variance
                    if g_t_x <= 0:
                        g_t_x = 0
                        alpha_t_x = g_t_x
                    else:
                        alpha_t_x = g_t_x
                    assert alpha_t_x >= 0
                x_vector = history_x.repeat(test_x.shape[0],1,1).to(device)
                x_vector = torch.cat((x_vector, test_x), 1)
                covar_query_x_and_x_vector = kernel(test_x, x_vector).evaluate()
                covar_x_vector_and_x_vector = kernel(x_vector, x_vector).evaluate()
                eye_matrix = torch.eye(covar_x_vector_and_x_vector.shape[-1]).to(device)
                noise = GP.likelihood.noise
                B = torch.linalg.inv(covar_x_vector_and_x_vector+noise*eye_matrix)
                coefficient_matrix = torch.matmul(covar_query_x_and_x_vector, B)
                coeff_y = history_y.repeat(test_x.shape[0],1,1)
                B = torch.matmul(coefficient_matrix[:,:,:-1], coeff_y )
                A = coefficient_matrix[:,:,-1]

                sigma = kernel(test_x, test_x).evaluate() - torch.matmul(coefficient_matrix, torch.transpose(covar_query_x_and_x_vector, 2, 1))
                if sigma <= 0:
                    sigma = torch.ones_like(sigma) * 0.0001
                B = torch.squeeze(B, 1)
                sigma = torch.sqrt(torch.squeeze(sigma, 1))
                posterior = GP.posterior(test_x, observation_noise = True)
                mean = posterior.mean
                variance = posterior.variance
                sigma_y = torch.sqrt(variance)
                mean_y = torch.squeeze(mean, 1)
                if iteration == 1:
                    upper = mean + sigma_y*torch.special.erfinv(torch.tensor(1-desired_alpha))*1.4142
                    lower = mean - sigma_y*torch.special.erfinv(torch.tensor(1-desired_alpha))*1.4142
                    upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
                else:
                    if alpha_t_x > 0:
                        upper = mean + sigma_y * torch.special.erfinv(
                            torch.tensor(1 - alpha_t_x).clone().detach()) * 1.4142
                        lower = mean - sigma_y * torch.special.erfinv(
                            torch.tensor(1 - alpha_t_x).clone().detach()) * 1.4142
                        upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
                    else:
                        upper, lower = (torch.sqrt(post_var) * 3.4 + post_mean), (
                                    -torch.sqrt(post_var) * 3.4 + post_mean)
                        upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
                y_star = acq_input_best_value
                sigma_y = torch.squeeze(sigma_y, 1)
                result_middle1, _ = integrate.quad(integrand_middle, y_star, np.inf, args = (A[0,0].detach().cpu().numpy(), B[0,0].detach().cpu().numpy(), sigma[0,0].detach().cpu().numpy(), lower[0,0].detach().cpu().numpy(), upper[0,0].detach().cpu().numpy(),sigma_y[0,0].detach().cpu().numpy(),y_star,alpha_t_x, desired_alpha))
                if alpha_t_x>0:
                    result_tale1, _ = integrate.quad(integrand_tale, y_star, np.inf, args = (A[0,0].detach().cpu().numpy(), B[0,0].detach().cpu().numpy(), sigma[0,0].detach().cpu().numpy(), lower[0,0].detach().cpu().numpy(), upper[0,0].detach().cpu().numpy(), sigma_y[0,0].detach().cpu().numpy(), alpha_t_x, mean_y[0,0].detach().cpu().numpy(),y_star, desired_alpha))
                else:
                    result_tale1 = 0
                ei = [result_middle1+result_tale1]
                ei_list = torch.tensor(ei).to(train_x)
        else:
            with torch.no_grad():
                c_t = desired_alpha
                test_x = torch.tensor(np.expand_dims(best_train_x[repeat_iter*j:repeat_iter*j+repeat_iter,:], axis=1)).to(device)
                if iteration == 1:
                    g_t_x = c_t
                    alpha_t_x = g_t_x
                else:
                    t_values = np.arange(1, iteration)
                    eta_list = eta_init / (t_values + 1) ** decay
                    eta_func_list = eta_init / (t_values + 1) ** decay
                    array1 = np.array(eta_list)
                    array2 = np.array([desired_alpha - x for x in err_list])
                    c_t = np.sum(array1 * array2) + c_t
                    lambda_term_list = []
                    kernel_ARC_list = []
                    array1 = np.array(eta_func_list)
                    array2 = np.array([desired_alpha - x for x in err_list])
                    eta_term_list = array1 * array2
                    for i in range(1, iteration):
                        if i == 1:
                            cum_prod = eta_list[-(i - 1)] ** 0
                        else:
                            cum_prod = (1 - reg_param * eta_list[-(i - 1)]) * cum_prod
                        lambda_term_list.append(cum_prod)
                        kernel_ARC_list.append(torch.exp(-torch.norm(history_x[n_init + i - 1] - test_x).to('cpu') ** 2 / (length_scale ** 2)))
                    lambda_term_list = lambda_term_list[::-1]
                    lambda_term_list = np.array(lambda_term_list)
                    kernel_ARC_list = np.array(kernel_ARC_list)
                    g_list = lambda_term_list * eta_term_list * kernel_ARC_list
                    g_t_x = eta_func_const * kappa * np.sum(g_list) + c_t
                    posterior = model.posterior(test_x, observation_noise=True)
                    post_mean, post_var = posterior.mean, posterior.variance
                    if g_t_x <= 0:
                        g_t_x = 0
                        alpha_t_x = g_t_x
                    else:
                        alpha_t_x = g_t_x
                    assert alpha_t_x >= 0
                x_vector = history_x.repeat(test_x.shape[0],1,1).to(device)
                x_vector = torch.cat((x_vector, test_x), 1)

                covar_query_x_and_x_vector = kernel(test_x, x_vector).evaluate()
                covar_x_vector_and_x_vector = kernel(x_vector, x_vector).evaluate()
                eye_matrix = torch.eye(covar_x_vector_and_x_vector.shape[-1]).to(device)
                noise = GP.likelihood.noise
                B = torch.linalg.inv(covar_x_vector_and_x_vector+noise*eye_matrix)
                coefficient_matrix = torch.matmul(covar_query_x_and_x_vector, B)
                coeff_y  = history_y.repeat(test_x.shape[0],1,1)
                B = torch.matmul(coefficient_matrix[:,:,:-1],coeff_y )
                A = coefficient_matrix[:,:,-1]
                sigma = kernel(test_x, test_x).evaluate() - torch.matmul(coefficient_matrix, torch.transpose(covar_query_x_and_x_vector, 2, 1))
                if sigma <= 0:
                    sigma = torch.ones_like(sigma) * 0.0001
                B = torch.squeeze(B, 1)
                sigma = torch.sqrt(torch.squeeze(sigma, 1))
                posterior = GP.posterior(test_x, observation_noise = True)
                mean = posterior.mean
                variance = posterior.variance
                mean_y = torch.squeeze(mean, 1)
                sigma_y = torch.sqrt(variance)
                upper = mean + sigma_y*torch.special.erfinv(torch.tensor(1-alpha_t_x))*1.4142
                lower = mean - sigma_y*torch.special.erfinv(torch.tensor(1-alpha_t_x))*1.4142
                upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
                sigma_y = torch.squeeze(sigma_y, 1)
                y_star = acq_input_best_value
                result_middle1, _ = integrate.quad(integrand_middle, y_star, np.inf, args = (A[0,0].detach().cpu().numpy(), B[0,0].detach().cpu().numpy(), sigma[0,0].detach().cpu().numpy(), lower[0,0].detach().cpu().numpy(), upper[0,0].detach().cpu().numpy(), sigma_y[0,0].detach().cpu().numpy(), y_star, alpha_t_x, desired_alpha))
                if alpha_t_x>0:
                    result_tale1, _ = integrate.quad(integrand_tale, y_star, np.inf, args = (A[0,0].detach().cpu().numpy(), B[0,0].detach().cpu().numpy(), sigma[0,0].detach().cpu().numpy(), lower[0,0].detach().cpu().numpy(), upper[0,0].detach().cpu().numpy(), sigma_y[0,0].detach().cpu().numpy(), alpha_t_x, mean_y[0,0].detach().cpu().numpy(), y_star, desired_alpha))
                else:
                    result_tale1 = 0
                ei = [result_middle1+result_tale1]
                ei = torch.tensor(ei).to(train_x)
                ei_list = torch.cat((ei_list,ei),axis = 0)
    best_x_idx = ei_list.argmax().item()
    new_x = best_train_x[best_x_idx,:]
    np.random.seed()
    exact_obj = O_1.observation_w_N_channels(new_x.reshape(len(BSs)*2,-1), fixed_dim*5 + num_channels, fixed_dim*5 + num_channels*2)
    new_obj = torch.tensor([exact_obj],dtype = torch.float32,device=device).unsqueeze(-1)
    return torch.tensor(new_x).view(-1,len(BSs)*2).to(device), new_obj

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
with warnings.catch_warnings():
  warnings.simplefilter("ignore", category=Warning)
  true_fn_list = []
  thermal_noise = 10**(-1*thermal_noise_arg)
  O_1 = objective( BSs, UEs, UAVs, thermal_noise, fading_l, angle_vertical, fixed_horz_angle)
  N_TRIALS = 10 if not SMOKE_TEST else 1
  N_BATCH = 250 if not SMOKE_TEST else 2
  num_channels = number_channels
  init_points = 50
  num_cand_acq = 100
  MC_SAMPLES = 256 if not SMOKE_TEST else 32
  desired_alpha = 0.45
  alpha_zero = desired_alpha
  alpha_rate = 0.005
  best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []
  temp = 3e-2
  grid_res=64
  max_grid_refinements = 0
  ratio_estimator = None
  print('num_channel', num_channels)
  err_list = []
  eta_init, reg_param, length_scale, kappa, decay = 0.005, 4e-3, 1/3, 1, 5e-3

  for trial in range(1, N_TRIALS + 1):
      saving_list = []
      print(f"\nTrial {trial} of {N_TRIALS} ", end="")
      best_observed_ei, best_observed_nei, best_random = [], [], []
      (
          train_x_ei,
          train_obj_ei,
          best_observed_value_ei,
      ) = cuav_generate_initial_data(num_init_points = init_points, num_channels=num_channels)
      mll_ei, model_ei, best_observe = initialize_model(train_x_ei, train_obj_ei)
      train_x_nei, train_obj_nei, = train_x_ei, train_obj_ei
      best_observed_value_nei = best_observed_value_ei

      best_observed_ei.append(best_observed_value_ei)
      best_random.append(best_observed_value_ei)
      best_object_ei = []
      after_BO_train_obj_ei = torch.tensor([]).to(best_observe)
      acq_input_best_value = -100
      printing_observed_ei = []
      alpha_list = []
      err_t_count = []
      err_list = []
      for iteration in range(1, N_BATCH + 1):
          if iteration%10 == 0:
            print(f"\niteration {iteration}  ", end="")
          if switch_fit_model == 1:
            if iteration % 10 == 0:
              fit_gpytorch_mll(mll_ei)
          else:
            if iteration ==1:
              fit_gpytorch_mll(mll_ei)
          mll_ei.eval()
          mll_ei.requires_grad_(False)
          qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

          qEI = qExpectedImprovement(
              model=model_ei,
              best_f=(train_obj_ei).max(),
              sampler=qmc_sampler,
          )
          EI = ExpectedImprovement(model_ei, best_f=acq_input_best_value)

          parms = {'best_f': acq_input_best_value, 'alpha': 0.2, 'grid_res': 64, 'max_grid_refinements': 0, 'temp': 0.01, 'randomized': True, 'ratio_estimator':None}

          t0 = time.monotonic()

          if iteration ==1:
            Tx_power = torch.ones((1,len(BSs)))
            tilting_angle = torch.ones((1,len(BSs)))*0.5
            next_query_ei = torch.concatenate((Tx_power, tilting_angle), axis = 1)
            alpha_t = alpha_zero
          new_x_ei, new_obj_ei = online_cbo_fixed_dim_get_observation(num_channels, next_query_ei.view(len(BSs)*2,-1).cpu().numpy() ,iteration, num_cand_acq, device, model_ei, train_x_ei, train_obj_ei, acq_input_best_value,
                                                                      desired_alpha, init_points, iteration, err_list, length_scale, eta_init, reg_param, kappa, decay)
          train_x_ei = torch.cat([train_x_ei, new_x_ei])
          train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
          print('new_obj_ei',new_obj_ei)

          if iteration != 1:
            if new_obj_ei >  after_BO_train_obj_ei.max():
              acq_input_best_value = standardize(train_obj_ei)[-1]
              next_query_ei = new_x_ei
              after_BO_train_obj_ei = torch.cat([after_BO_train_obj_ei, new_obj_ei])
            else:
              acq_input_best_value = standardize(train_obj_ei)[init_points+after_BO_train_obj_ei.argmax()]
              next_query_ei = next_query_ei
              after_BO_train_obj_ei = torch.cat([after_BO_train_obj_ei, new_obj_ei])
            parms = {'best_f': best_observe, 'alpha': alpha_t, 'grid_res': 64, 'max_grid_refinements': 0, 'temp': 0.01, 'randomized': True, 'ratio_estimator': None}
          else:
            acq_input_best_value = standardize(train_obj_ei)[-1]
            next_query_ei = new_x_ei
            after_BO_train_obj_ei = torch.cat([after_BO_train_obj_ei, new_obj_ei])
            parms = {'best_f': best_observe, 'alpha': alpha_zero, 'grid_res': 64, 'max_grid_refinements': 0, 'temp': 0.01, 'randomized': True, 'ratio_estimator': None}
            alpha_t = alpha_zero
          best_query_after_initialize = train_x_ei[after_BO_train_obj_ei.argmax()+init_points]
          printing_observed_ei.append(after_BO_train_obj_ei.max().item())
          t1 = time.monotonic()

          mll_ei, model_ei, best_observe = initialize_model(
              train_x_ei,
              train_obj_ei,
              model_ei.state_dict(),
          )
          alpha_list.append(alpha_t)
          err_list = online_cbo_alpha(train_x_ei, standardize(train_obj_ei)[-1], alpha_zero, model_ei, iteration,
                                      eta_init, reg_param, length_scale,
                                      decay, err_list, init_points)
          print('err_t',sum(err_list))
          print( f"time = {t1-t0:>4.2f}.")
          torch.cuda.empty_cache()
          print(".", end="")
          if iteration % 25 == 0:
              true_fn = O_1.MC_true_fun(best_query_after_initialize.view(len(BSs) * 2, -1).cpu().numpy())
              saving_list.append(true_fn)
      if trial == 1:
          save_list = np.array(saving_list)
      else:
          arr_saving_list = np.array(saving_list)
          save_list = np.vstack((save_list, arr_saving_list))
      np.save('./history/LOCBO_CUAV_' + str(num_channels) +'length_' + '033_alpha_035', save_list)
      best_observed_all_ei.append(best_observed_ei)
      true_fn = O_1.MC_true_fun(best_query_after_initialize.view(len(BSs)*2,-1).cpu().numpy())
      true_fn_list.append(true_fn)
      print('save size', save_list.shape)
      print('num channels/length_scale',num_channels,length_scale)
      for num in true_fn_list:
        rounded_num = round(num, 2)
        print(rounded_num, end=' ')

print(true_fn_list)
print('num channels',num_channels)
for num in true_fn_list:
    rounded_num = round(num, 2)
    print(rounded_num, end=' ')
