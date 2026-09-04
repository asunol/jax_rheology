import diff_rheo as dr
import jax
import jax.numpy as jnp
import json
import equinox as eqx
from scipy.stats import chi2
import argparse
from diffrax import DirectAdjoint
import time
import os

# jax.config.update("jax_enable_x64", True)

model_list = ["Newtonian", "CarreauYasuda", "OldroydB", "Giesekus", "LinearPTT"]
# model_list = ["CarreauYasuda"]
model_lookup = {
    "Newtonian": dr.models.Newtonian,
    "CarreauYasuda": dr.models.CarreauYasuda,
    "OldroydB": dr.models.OldroydB,
    "Giesekus": dr.models.Giesekus,
    "LinearPTT": dr.models.LinearPTT,
    "FENECR": dr.models.FENECR
}

parameter_lookup = {
    "Newtonian": {
        "viscosity": (1e-1,1e1,"log"),
    },
    "CarreauYasuda": {
        "zero_shear_viscosity": (1.0,1e2,"log"),
        "infinite_shear_viscosity": (1e-2,.1,"log"),
        "k": (1e-1,1e1,"log"),
        "n": (0.2,.7,"uniform"),
        "a": (0.5,3.0,"uniform"),
    },
    "OldroydB": {
        "polymer_viscosity": (1.0,10.0,"uniform"),
        "relaxation_time": (1.0,10.0,"uniform"),
        "solvent_viscosity": (0.1,10.0,"uniform"),
    },
    "Giesekus": {
        "polymer_viscosity": (1e-1,1e1,"log"),
        "relaxation_time": (1.0,10.0,"log"),
        "solvent_viscosity": (1e-1,1e1,"log"),
        "alpha": (0.01,.5,"uniform"),
    },
    "LinearPTT": {
        "polymer_viscosity": (1e-1,1e1,"log"),
        "relaxation_time": (1.0,10.0,"log"),
        "solvent_viscosity": (1e-1,1e1,"log"),
        "zeta": (0.01,.2,"uniform"),
        "epsilon": (0.01,.5,"uniform"),
    },
    "FENECR": {
        "polymer_viscosity": (1e-1,1e1,"log"),
        "relaxation_time": (1.0,10.0,"log"),
        "solvent_viscosity": (1e-1,1e1,"log"),
        "extension_length": (1.0,10.0,"log"),
    },
}

@eqx.filter_jit
def _run_single_experiment(forcing_data, model, key, rheometer, initial_condition,time_range):
    def vel_func(t: jax.Array) -> jax.Array:
        gamma_dot, omega = forcing_data
        return gamma_dot * jnp.sin(omega * t)
    
    vel_func_forcing = dr.VelocityGradient.from_components(grad_u_12=vel_func)
    vel_data = vel_func(time_range) 
    data = rheometer.run_experiment(model,vel_func_forcing, time_range, initial_condition)
    shear_stress = data.data[:,0,1]
    
    noise_level = model.observation_noise.get_value()
    noise = jax.random.normal(key, shape=shear_stress.shape) * noise_level

    return dr.ShearStrainRateData(
        time=time_range,
        data=shear_stress + noise,
        initial_condition = initial_condition,
        forcing_data=vel_data
    )

def generate_ground_truth(model,noise_level: float, key: jax.random.PRNGKey)-> dr.BatchedData:
    solver = dr.DiffraxSolver()
    rheometer = dr.VirtualRheometer.setup(model, "strain_rate_response", solver)
    initial_condition = jnp.zeros((3,3))
    time_range = jnp.linspace(0.0, 12.0, 100)
    
    # params_list = []
    # for gammadot in [1.0,0.1,10.0,.01]:
    #     for omega in [1/3., 1., 2.]:
    #         params_list.append([gammadot, omega])
    
    params_list = [[1.0,1.0]]
    stacked_params = jnp.array(params_list)
    num_experiments = len(stacked_params)
    keys = jax.random.split(key, num_experiments)

    vmapped_runner = eqx.filter_vmap(
        _run_single_experiment,
        in_axes=(0, None, 0, None, None, None)
    )

    datasets = vmapped_runner(
        stacked_params, model, keys, rheometer, initial_condition, time_range
    )
  
    batch_size = datasets.data.shape[0]

    unbatched_datasets = [
        dr.ShearStrainRateData(
            time=datasets.time[i],
            data=datasets.data[i],
            initial_condition=datasets.initial_condition[i],
            forcing_data=datasets.forcing_data[i]
        )
        for i in range(batch_size)
    ]

    return dr.BatchedData.from_data(*unbatched_datasets)

def initialize_guess_models():
    guess_models = {}
    for model_name in model_list:
        model = model_lookup[model_name]
        params = {}
        for name in parameter_lookup[model_name].keys():
            params[name] = dr.parameters.LogParameter(1.0)
            if model_name == "CarreauYasuda":
                params["zero_shear_viscosity"] = dr.parameters.LogParameter(15.0)
                params["infinite_shear_viscosity"] = dr.parameters.LogParameter(0.01)
                params["n"] = dr.parameters.LogParameter(0.2)
                params["k"] = dr.parameters.LogParameter(4.0)
                params["a"] = dr.parameters.LogParameter(2.0)
            if model_name == "Giesekus":
                params["alpha"] = dr.parameters.TanhParameter(0.1)
            if model_name == "LinearPTT":
                params["zeta"] = dr.parameters.LogParameter(0.1)
                params["epsilon"] = dr.parameters.LogParameter(0.1)
            if model_name == "FENECR":
                params["extension_length"] = dr.parameters.LogParameter(5.0)
        params["observation_noise"] = dr.parameters.LogParameter(1.0)
        model = model(**params)
        guess_models[model_name] = model
    return guess_models

def generate_random_model(key: jax.random.PRNGKey, model_idx: int = None):
    key, subkey = jax.random.split(key)
    num_models = len(model_list)
    
    if model_idx is None:
        model_idx = jax.random.choice(subkey, num_models)
    else:
        if model_idx < 0 or model_idx >= num_models:
            raise ValueError(f"Model index {model_idx} is out of range. Valid indices: 0-{num_models-1}")
    
    model_name = model_list[model_idx]
    model = model_lookup[model_name]
    params = {}
    for name, (min_val, max_val, dist_type) in parameter_lookup[model_name].items():
        key, subkey = jax.random.split(key)
        if dist_type == "log":
            # Log uniform distribution
            log_min = jnp.log10(min_val)
            log_max = jnp.log10(max_val)
            log_value = jax.random.uniform(subkey, minval=log_min, maxval=log_max)
            params[name] = jnp.power(10, log_value)
        else:  # uniform
            params[name] = jax.random.uniform(subkey, minval=min_val, maxval=max_val)
    model = model(**params)
    return model, model_name

def fit_random_model(model, exp_data, key: jax.random.PRNGKey, config: dr.FittingConfig):
    key, subkey = jax.random.split(key)
    exp_data = generate_ground_truth(model, 0.00, subkey)
    print(f"Generated {len(exp_data.data)} experiments", flush=True)
    guess_models = initialize_guess_models()
    print(f"Initialized {len(guess_models)} guess models", flush=True)
    solver = dr.DiffraxSolver(max_steps=1000, throw=False)
    results = {}
    bic_vals = {}
    for guess_model_name, guess_model in guess_models.items():
        print(f"Fitting model: {guess_model_name}", flush=True)
        rheometer = dr.VirtualRheometer.setup(guess_model, "strain_rate_response", solver)
        try:
            fit_variational_model = dr.fit_model_to_experimental_data(guess_model, rheometer, exp_data, config)
            # bic = dr.model_bic(model, rheometer, exp_data, config, direct=True)
            bic = dr.calculate_bic_from_l2(fit_variational_model, rheometer, exp_data)
            print(f"Model: {guess_model_name} BIC: {bic:.2f}", flush=True)
            results[guess_model_name] = {
                "bic": bic,
                "parameter_values": fit_variational_model.parameter_values,
            }
        except Exception as e:
            print(f"Error fitting model: {guess_model_name}", flush=True)
            results[guess_model_name] = {
                "bic": jnp.inf,
                "parameter_values": None,
            }
    return results

def compare_models(results: dict, model_name: str, params: dict, exp_data: dr.BatchedData, config: dr.FittingConfig):
    best_model_name = min(results.keys(), key=lambda name: results[name]["bic"])
    best_bic = results[best_model_name]["bic"]
    print(f"\nBest model: {best_model_name} with BIC: {best_bic:.2f}", flush=True)
    print(f"True model was: {model_name}", flush=True)
    print(f"Model selection {'correct' if best_model_name == model_name else 'incorrect'}", flush=True)
    print(f"True model parameters: {params}", flush=True)
    print(f"Best model parameters: {results[best_model_name]['parameter_values']}", flush=True)
    # uncertainties = get_asymptotic_uncertainties_with_names(best_model_name, results[best_model_name]['parameter_values'], exp_data, config)
    return best_model_name, best_bic, results[best_model_name]['parameter_values']

def random_test_loop(key: jax.random.PRNGKey, config: dr.FittingConfig, model_idx: int = None):
    key, subkey = jax.random.split(key)
    model, model_name = generate_random_model(subkey, model_idx)
    print(f"Generated random model: {model_name}", flush=True)
    print(f"Model parameters: {model.parameter_values}", flush=True)
    exp_data = generate_ground_truth(model, 0.03, subkey)
    results = fit_random_model(model, exp_data, key, config)
    best_model_name, best_bic, best_params = compare_models(results, model_name, model.parameter_values, exp_data, config)
    summary = {
        "model_name": model_name,
        "model_parameters": model.parameter_values,
        "best_model_name": best_model_name,
        "best_bic": best_bic,
        "best_params": best_params,
    }
    bic_record = {
        "true_model_name": model_name,
        "true_model_parameters": model.parameter_values,
        "bics": {name: results[name]["bic"] for name in results},
    }
    return summary, bic_record

def convert_jax_to_python(obj):
        """Recursively convert JAX arrays to Python floats/lists"""
        if hasattr(obj, 'item'):  # JAX scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # JAX array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_jax_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_jax_to_python(item) for item in obj]
        else:
            return obj
def export_to_json(results: dict, filename: str):
    results = convert_jax_to_python(results)
    with open(filename, 'a') as f:
        json.dump(results, f)
        f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random data and fit models")
    parser.add_argument("--model_idx", type=int, default=None, 
                       help=f"Model index to use (0-{len(model_list)-1}). If not provided, randomly selects a model.")
    parser.add_argument("--num_runs", type=int, default=2,
                       help="Number of runs to perform (default: 2)")
    parser.add_argument("--seed", type=int, default=None,
                       help="PRNG seed (default: current UTC timestamp).")

    results_dir = os.path.join(os.path.dirname(__file__), "results_synthetic")
    args = parser.parse_args()
    # Create results_l2 directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    if args.seed is not None:
        utc_timestamp = args.seed
        print(f"Using user-supplied seed: {utc_timestamp}", flush=True)
    else:
        utc_timestamp = int(time.time())
        print(f"Using UTC timestamp as seed: {utc_timestamp}", flush=True)
    key = jax.random.PRNGKey(utc_timestamp)
    key, subkey = jax.random.split(key)

    if args.model_idx is not None:
        print(f"Using model index: {args.model_idx} ({model_list[args.model_idx]})", flush=True)
        results_file = f"{results_dir}/results_l2_{model_list[args.model_idx]}.json"
        bics_file = f"{results_dir}/bics_l2_{model_list[args.model_idx]}.json"
    else:
        print("Using random model selection", flush=True)
        results_file = f"{results_dir}/results_l2_random.json"
        bics_file = f"{results_dir}/bics_l2_random.json"
    with open(results_file, 'w') as f:
        pass
    with open(bics_file, 'w') as f:
        pass
    n_epochs = 1000
    learning_rate = 1e-1
    noise_level = 0.03
    schedule_lr = False
    with open(results_file, 'a') as f:
        config_data = {
            "num_runs": args.num_runs,
            "model_idx": args.model_idx,
            "model_list": model_list,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "seed": utc_timestamp,
            "parameter_lookup": parameter_lookup,
            "noise_level": noise_level,
            "schedule_lr": schedule_lr,
        }
        json.dump(config_data, f)
        f.write('\n')
    for i in range(args.num_runs):
        print(f"Running run {i+1} of {args.num_runs}", flush=True)
        key, subkey = jax.random.split(key)
        config = dr.FittingConfig(
            num_epochs=n_epochs,
            learning_rate=learning_rate,
            ensemble_size=100,
            key = subkey,
            verbose=True,
            schedule_lr=schedule_lr,
        )
        key, subkey = jax.random.split(key)
        result, bic_record = random_test_loop(subkey, config, args.model_idx)
        export_to_json(result, results_file)
        export_to_json(bic_record, bics_file)

    print("Evaluation complete", flush=True)

