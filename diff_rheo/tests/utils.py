from pathlib import Path
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma
from diff_rheo._utils import _generalized_mittag_leffler_function

def parse_analytical_solutions(filepath: Path) -> dict:
    """
    Parses the text file containing analytical solutions derived from Mathematica.
    """
    solutions = {}
    current_model = None
    current_test_type = None
    
    # Keywords that separate the model name from the test type in the header
    test_type_keywords = ["extra", "strain"]

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                parts = line[1:].strip().split(' ')
                
                # Find where the model name ends and the test type begins
                split_index = -1
                for i, part in enumerate(parts):
                    if part in test_type_keywords:
                        split_index = i
                        break
                
                if split_index != -1:
                    model_name_parts = parts[:split_index]
                    test_type_parts = parts[split_index:]
                    
                    # Join parts to form the full model name and test type
                    model_name = '_'.join(model_name_parts).replace('-', '_')
                    test_type = '_'.join(test_type_parts)

                    if model_name not in solutions:
                        solutions[model_name] = {}
                    
                    current_model = model_name
                    current_test_type = test_type
                    solutions[current_model][current_test_type] = {}
            
            elif '=' in line and current_model:
                var, expr = line.split('=', 1)
                solutions[current_model][current_test_type][var] = expr
                
    return solutions

def general_ptt_f_ptt(extra_stress_matrix: jax.Array, model_params: dict) -> jax.Array:
    alpha = model_params["alpha"]
    beta = model_params["beta"]
    epsilon = model_params["epsilon"]
    relaxation_time = model_params["relaxation_time"]
    polymer_viscosity = model_params["polymer_viscosity"]

    normalization = gamma(beta)
    argument = (epsilon*relaxation_time/polymer_viscosity * jnp.trace(extra_stress_matrix), alpha, beta)
    mittag_leffler = _generalized_mittag_leffler_function(*argument)
    f_ptt = normalization * mittag_leffler
    return f_ptt