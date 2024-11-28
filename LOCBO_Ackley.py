from LOCBO_ackley_imports import *

def integrand_torch_middle(x, a, b, sigma, l, u, sigma_y, best_f, alpha, desired_alpha):
    x = x.to(a)
    val_2 = torch.tensor(2.0)
    return (1 - desired_alpha) * 1 / (u - l) * 1 / (2 * a) * (
                torch.erf(1 / (torch.sqrt(val_2).to(a) * sigma) * (-x + a * u + b)) - torch.erf(
            1 / (torch.sqrt(val_2).to(a) * sigma) * (-x + a * l + b))) * torch.max(x - best_f, torch.tensor([0]).to(x))

def integrand_torch_tale(x, a, b, sigma, l, u, sigma_y, alpha, mu_y, best_f, desired_alpha):
    x = x.to(a)
    val_pi = torch.tensor(torch.pi)
    A_square = (a ** 2 / (2 * sigma ** 2)) ** 2
    B = (x - b) / (a)
    C_square = (1 / (2 * sigma_y ** 2)) ** 2
    D = mu_y
    upper = 1 / (2 * val_pi * sigma * sigma_y) * 0.5 * torch.sqrt(val_pi / (A_square + C_square)) * torch.exp \
        (-A_square * C_square / (A_square + C_square) * (B - D) ** 2) * (1 - torch.erf(
        (-B * A_square - D * C_square + (A_square + C_square) * u) / torch.sqrt(A_square + C_square)))
    lower = 1 / (2 * val_pi * sigma * sigma_y) * 0.5 * torch.sqrt(val_pi / (A_square + C_square)) * torch.exp(
        -A_square * C_square / (A_square + C_square) * (B - D) ** 2) * (1 + torch.erf(
        (-B * A_square - D * C_square + (A_square + C_square) * l) / torch.sqrt(A_square + C_square)))
    return desired_alpha / alpha * (upper + lower) * torch.max(x - best_f, torch.tensor([0]).to(x))

class qScalarizedUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            desired_alpha: Tensor,
            num_init: Tensor,
            MC_samples_integration: Tensor,
            best_f: Tensor,
            weights: Tensor,
            iteration: Tensor,
            err_list: Tensor,
            length_scale: Tensor,
            kappa: Tensor,
            eta_0: Tensor,
            decay: Tensor,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("desired_alpha", torch.as_tensor(desired_alpha))
        self.register_buffer("num_init", torch.as_tensor(num_init))
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("weights", torch.as_tensor(weights))
        self.register_buffer("iteration", torch.as_tensor(iteration))
        self.register_buffer("err_list", torch.as_tensor(err_list))
        self.register_buffer("length_scale", torch.as_tensor(length_scale))
        self.register_buffer("kappa", torch.as_tensor(kappa))
        self.register_buffer("eta_0", torch.as_tensor(eta_0))
        self.register_buffer("decay", torch.as_tensor(decay))
        self.register_buffer("MC_samples_integration", torch.as_tensor(MC_samples_integration))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        history_x = self.model.train_inputs[0]
        history_y = torch.unsqueeze(standardize(self.model.train_targets), dim=1)
        test_x = X.to(history_x)  # [2,1,114]/ [1,1,1]
        eta_init, reg_param, length_scale, kappa, decay = self.eta_0, 1e-4, self.length_scale, self.kappa, self.decay
        c_t, iteration = self.desired_alpha, self.iteration
        eta_func_const = 1*kappa
        if iteration == 1:
            g_t_x = c_t
            alpha_t_x = g_t_x
        else:
            t_values = np.arange(1, iteration)
            eta_list = eta_init / (t_values + 1) ** decay.cpu().numpy()
            eta_func_list = eta_init / (t_values + 1) ** decay.cpu().numpy()
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
                kernel_ARC_list.append(torch.exp(-torch.norm(history_x[self.num_init.item()+i - 1].to('cpu') - test_x.to('cpu')) ** 2 / (length_scale ** 2)).detach().numpy())
            lambda_term_list = lambda_term_list[::-1]
            lambda_term_list = np.array(lambda_term_list)
            kernel_ARC_list = np.array(kernel_ARC_list)
            g_list = lambda_term_list * eta_term_list * kernel_ARC_list
            g_t_x = eta_func_const * kappa * np.sum(g_list) + c_t
            posterior = self.model.posterior(test_x, observation_noise=True)
            post_mean, post_var = posterior.mean, posterior.variance
            if g_t_x<=0:
                g_t_x = 0
                alpha_t_x = g_t_x
                upper = torch.sqrt(post_var)*3.4 + post_mean
                lower = -torch.sqrt(post_var)*3.4 + post_mean
            else:
                upper = norm.ppf(1 - g_t_x / 2) * torch.sqrt(post_var) + post_mean
                lower = norm.ppf(g_t_x / 2) * torch.sqrt(post_var) + post_mean
                alpha_t_x = g_t_x
        assert alpha_t_x >= 0
        kernel = self.model.covar_module
        x_vector = history_x.repeat(test_x.shape[0], 1, 1).to(device)  # [], [1,1,1]
        x_vector = torch.cat((x_vector, test_x), 1)  # [2,51,114], [1,2,1]
        covar_query_x_and_x_vector = kernel(test_x, x_vector).evaluate()  # [2,1,51], [1,1,2]
        covar_x_vector_and_x_vector = kernel(x_vector, x_vector).evaluate()  # [2,51,51], [1,2,2]
        eye_matrix = torch.eye(covar_x_vector_and_x_vector.shape[-1]).to(device)  # [51,51], [2,2]
        noise = self.model.likelihood.noise  # noise variance return
        B = torch.linalg.inv(covar_x_vector_and_x_vector + noise * eye_matrix)
        coefficient_matrix = torch.matmul(covar_query_x_and_x_vector, B)  # [2,1,51]
        coeff_y = history_y.repeat(test_x.shape[0], 1, 1)  # [2,50,1]

        B = torch.matmul(coefficient_matrix[:, :, :-1], coeff_y)
        A = coefficient_matrix[:, :, -1]
        # Mean = ay'+b
        sigma_square = kernel(test_x, test_x).evaluate() - torch.matmul(coefficient_matrix, torch.transpose(covar_query_x_and_x_vector, 2, 1))
        B = torch.squeeze(B, 1)
        sigma = torch.sqrt(torch.squeeze(sigma_square, 1))
        posterior = self.model.posterior(test_x, observation_noise=True)
        mean = posterior.mean
        variance = posterior.variance
        sigma_y = torch.sqrt(variance)
        mean_y = torch.squeeze(mean, 1)
        if self.iteration == 1:
            upper = mean + sigma_y * torch.special.erfinv(
                torch.tensor(1 - self.desired_alpha).clone().detach()) * 1.4142
            lower = mean - sigma_y * torch.special.erfinv(
                torch.tensor(1 - self.desired_alpha).clone().detach()) * 1.4142
            upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
        else:
            if alpha_t_x >0:
                upper = mean + sigma_y * torch.special.erfinv(torch.tensor(1 - alpha_t_x).clone().detach()) * 1.4142
                lower = mean - sigma_y * torch.special.erfinv(torch.tensor(1 - alpha_t_x).clone().detach()) * 1.4142
                upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
            else:
                upper, lower = (torch.sqrt(post_var)*3.4 + post_mean),  (-torch.sqrt(post_var)*3.4 + post_mean)
                upper, lower = torch.squeeze(upper, 1), torch.squeeze(lower, 1)
        y_star = self.best_f
        sigma_y = torch.squeeze(sigma_y, 1)
        input_sigma = sigma
        input_sigma_y = sigma_y

        def int_torch_middle(x):
            a, b, sigma, l, u, sigma_y, best_f = A, B, input_sigma, lower, upper, input_sigma_y, y_star # Example value for 'a'
            alpha = self.desired_alpha if self.iteration == 1 else alpha_t_x
            desired_alpha = self.desired_alpha
            return integrand_torch_middle(x, a, b, sigma, l, u, sigma_y, best_f, alpha, desired_alpha)

        def int_torch_tale(x):
            a, b, sigma, l, u, sigma_y, mu_y, best_f = A, B, input_sigma, lower, upper, input_sigma_y, mean_y, y_star  # Example value for 'a'
            alpha = self.desired_alpha if self.iteration == 1 else alpha_t_x
            desired_alpha = self.desired_alpha
            return integrand_torch_tale(x, a, b, sigma, l, u, sigma_y, alpha, mu_y, best_f, desired_alpha)
        mc = MonteCarlo()
        torch_middle_value = mc.integrate(
            int_torch_middle,
            dim=1,
            N=self.MC_samples_integration.item(),
            integration_domain=[[y_star, 5]],
            backend="torch",
        )
        torch_tale_value = (
            mc.integrate(
                int_torch_tale,
                dim=1,
                N=self.MC_samples_integration.item(),
                integration_domain=[[y_star, 5]],
                backend="torch",
            )
            if alpha_t_x != 0 else 0
        )
        integral_value = torch_middle_value + torch_tale_value
        assert integral_value>=0
        return torch.unsqueeze(integral_value, dim=0)

# hetero_switch, input_acq, input_ntrial, input_MC_samples, input_length, input_kappa = \
#     int(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.double
from botorch.test_functions import Ackley
syn_obj_fun = Ackley(dim=2, negate=True)

def true_obj(X):
    return syn_obj_fun(X)

def hetero_noise_init(x, evaluate_val, noise_switch, trial):
    noise = (torch.norm(x, dim=1)+10)/20
    noise = noise_switch * torch.normal(0, torch.sqrt(noise))
    torch.seed()
    return noise.unsqueeze(1) + evaluate_val

def hetero_noise(x, evaluate_val, noise_switch):
    noise = (torch.norm(x, dim=1)+10)/20
    noise = noise_switch * torch.normal(0, torch.sqrt(noise))
    torch.seed()
    return noise + evaluate_val

def generate_initial_data(n, noise_switch, trial):
    torch.manual_seed(3 * trial)
    range1 = 20 * (torch.rand(n, syn_obj_fun.dim, device=device, dtype=dtype) * 0.4) - 10  # Values in the range [0, 0.3)
    range2 = 20 * (torch.rand(n, syn_obj_fun.dim, device=device, dtype=dtype) * 0.4 + 0.6) - 10  # Values in the range [0.7, 10)
    combined = torch.cat((range1, range2), dim=0)
    indices = torch.randperm(combined.size(0))
    shuffled = combined[indices]
    train_x = shuffled[:n, :]
    exact_obj = true_obj((train_x)).unsqueeze(-1)  # add output dimension
    train_obj = hetero_noise_init((train_x), exact_obj, noise_switch, trial)
    best_observed_value = true_obj((train_x)).max().item()
    return train_x, train_obj, best_observed_value

def initialize_model(train_x, train_obj, state_dict=None):
    model_obj = SingleTaskGP(train_x, standardize(train_obj)).to(train_x)
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    return mll, model_obj

def online_cbo_alpha(train_x, train_obj, alpha_desired, model_ei, iteration, kappa, eta_init, reg_param, length_scale, decay,
                     err_list, n_init):
    test_x = torch.unsqueeze(train_x[-1], 0)
    posterior = model_ei.posterior(test_x, observation_noise=True)
    post_mean, post_var = posterior.mean, posterior.variance
    c_t = alpha_desired
    eta_func_const = kappa*1
    if iteration > 1:
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
            kernel_ARC_list.append(
                torch.exp(-torch.norm(train_x[n_init + i - 1] - test_x).to('cpu') ** 2 / (length_scale ** 2)))
        lambda_term_list = lambda_term_list[::-1]
        lambda_term_list = np.array(lambda_term_list)
        kernel_ARC_list = np.array(kernel_ARC_list)
        g_list = lambda_term_list * eta_term_list * kernel_ARC_list
        g_t_x = eta_func_const * kappa * np.sum(g_list) + c_t
    else:
        g_t_x = c_t


    g_t_x = max(g_t_x, 0)
    u = norm.ppf(1 - g_t_x / 2) * torch.sqrt(post_var) + post_mean
    l = norm.ppf(g_t_x / 2) * torch.sqrt(post_var) + post_mean
    print(f"c_t: {c_t:.3f}, g_t: {g_t_x:.3f}, g_t - c_t: {g_t_x - c_t:.3f}")
    err_t = 0 if l <= train_obj <= u else 1
    err_list.append(err_t)
    return err_list

bounds = torch.tensor([[-10.0] * syn_obj_fun.dim, [10.0] * syn_obj_fun.dim], device=device, dtype=dtype)
BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES = 1, 10, 512

def optimize_acqf_and_get_observation(acq_func, noise_switch):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 1, "maxiter": 200},
    )
    new_x = candidates.detach()
    exact_obj = true_obj((new_x)).unsqueeze(-1)  # add output dimension
    if noise_switch == True:
        print('noise')
    new_obj = hetero_noise((new_x), exact_obj, noise_switch)
    return new_x, new_obj, exact_obj

def update_random_observations(best_random):
    rand_x = torch.rand(BATCH_SIZE, 6)
    next_random_best = true_obj(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

args = parse_arguments()
parser = argparse.ArgumentParser(description="Process input arguments for the script.")
parser.add_argument("hetero_switch", type=int, default=1, help="Heteroscedasticity switch (int)")
parser.add_argument("input_acq", type=str, default='OCEI', help="Acquisition function (str)")
parser.add_argument("input_ntrial", type=int, default=7, help="Number of trials (int)")
parser.add_argument("input_MC_samples", type=int, default=64, help="Number of Monte Carlo samples (int)")
parser.add_argument("input_length", type=int, default=5, help="Length scale (int)")
parser.add_argument("input_kappa", type=int, default=2, help="Kappa value (int)")

args = parser.parse_args()

hetero_switch, input_acq, input_ntrial, input_MC_samples, input_length, input_kappa = args.hetero_switch, args.input_acq, args.input_ntrial, args.input_MC_samples, args.input_length, args.input_kappa
N_TRIALS, N_BATCH, MC_SAMPLES, n_init, desired_alpha = input_ntrial, 50, 256, 5, 0.2 # BO hyperparameter
best_observed_all_ei, best_observed_all_nei, best_true_obj, err_list = [], [], [], [] # List for saving data
eta_init, reg_param, length_scale, kappa, decay, length_scale_fit = 0.005, 4e-3, input_length, input_kappa, 0.05, 'none' #LOCBO kernel hyperparameter
for trial in range(1, N_TRIALS + 1):
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_ei, best_random = [], []
    (
        train_x_ei,
        train_obj_ei,
        best_observed_value_ei,
    ) = generate_initial_data(n_init, hetero_switch, trial)
    if hetero_switch == 0:
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
    best_observed_ei.append(true_obj((train_x_ei[train_obj_ei.argmax()])).item())
    alpha_zero = desired_alpha
    alpha_rate = 0.005
    err_list = []
    history_per_iter = []
    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()
        print('iteration', iteration)
        t0 = time.monotonic()
        fit_gpytorch_mll(mll_ei)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        OCEI = qScalarizedUpperConfidenceBound(model_ei, desired_alpha=desired_alpha, num_init=n_init,
                                               MC_samples_integration=input_MC_samples,
                                               best_f=(standardize(train_obj_ei)).max(),
                                               weights=torch.tensor([0.1, 0.5]), iteration=iteration, err_list=err_list, length_scale=length_scale, kappa = kappa, eta_0 = eta_init, decay = decay)
        print('--------------------')

        new_x_ei, new_obj_ei, true_obj_val = optimize_acqf_and_get_observation(OCEI, hetero_switch) if input_acq == 'OCEI' else (None, None, None)
        train_x_ei, train_obj_ei = torch.cat([train_x_ei, new_x_ei]), torch.cat([train_obj_ei, new_obj_ei])
        best_value_ei = true_obj((train_x_ei[train_obj_ei.argmax()])).item()
        print('new x:', ', '.join(f'{x:.4f}' for x in new_x_ei.flatten()))
        print(f'exact y: {true_obj((new_x_ei)).item():.4f}')
        print('new y:', ', '.join(f'{x:.4f}' for x in new_obj_ei.flatten()))
        print(f'best true y: {best_value_ei:.4f}')
        best_observed_ei.append(best_value_ei)
        err_list = online_cbo_alpha(train_x_ei, standardize(train_obj_ei)[-1], alpha_zero, model_ei, iteration, kappa,
                                    eta_init, reg_param, length_scale,
                                    decay, err_list, n_init)
        # print('err_list', err_list)
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            model_ei.state_dict(),
        )
        torch.cuda.empty_cache()
        t1 = time.monotonic()
        print(f'Running time: {t1 - t0:.4f}, Kappa/Eta_0: {kappa:.4f}/{eta_init:.4f}.', end=" ")

    base_path = './history/Ackley_'
    ocbo_type = 'OCBO' if kappa == 0 else 'LOCBO'
    noise_type = 'pure' if hetero_switch == 0 else 'noise'
    adj_suffix = '_adj' if kappa != 0 and length_scale_fit == 'adj' else ''
    file_name = f"{base_path}{ocbo_type}_{noise_type}{adj_suffix}{str(trial)}_{str(desired_alpha)}"
    file_name += f"_{str(length_scale)}" if kappa != 0 else ""

    np.save(file_name, np.array(best_observed_ei))
    best_true_obj.append(best_value_ei)
    print(round(best_value_ei, 2))
    print('alpha rate/length_scale/alpha', alpha_rate, length_scale, desired_alpha)
    best_observed_all_ei.append(best_observed_ei)
    for num in best_true_obj:
        rounded_num = round(num, 2)
        print(rounded_num, end=' ')



