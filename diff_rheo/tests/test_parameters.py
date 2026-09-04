import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from diff_rheo.parameters import (
    Parameter, StaticParameter, LogParameter, TanhParameter,
    GaussianParameter, LogGaussianParameter
)

@pytest.fixture
def param():
    """A standard trainable parameter."""
    return Parameter(5.0)

@pytest.fixture
def static_param():
    """A non-trainable static parameter."""
    return StaticParameter(5.0)

@pytest.fixture
def log_param():
    """A parameter with a log-exp transformation."""
    return LogParameter(10.0)

@pytest.fixture
def tanh_param():
    """
    A parameter constrained to (0, max_value).
    """
    return TanhParameter(0.8, max_value=1.0)

@pytest.fixture
def gaussian_param():
    """A stochastic parameter from a Gaussian distribution."""
    return GaussianParameter(mean=5.0, std=0.1)

@pytest.fixture
def log_gaussian_param():
    """A stochastic parameter from a Log-Gaussian distribution."""
    return LogGaussianParameter(mean=5.0, std=0.1)

def test_parameter_behavior(param):
    assert jnp.isclose(param.get_value(), 5.0)

def test_log_parameter_behavior(log_param):
    assert jnp.isclose(log_param.get_value(), 10.0)
    assert jnp.isclose(log_param.value, jnp.log(10.0))

def test_tanh_parameter_behavior(tanh_param):
    assert jnp.isclose(tanh_param.get_value(), 0.8, atol=1e-6)

@pytest.mark.parametrize("param_fixture_name", [
    "gaussian_param",
    "log_gaussian_param"
])
def test_stochastic_parameter_behavior(param_fixture_name, request, rng_key):
    stochastic_param = request.getfixturevalue(param_fixture_name)
    
    key1, key2 = jax.random.split(rng_key)
    assert stochastic_param.sample(key1).get_value() != stochastic_param.sample(key2).get_value()
    assert stochastic_param.sample(key1).get_value().shape == ()


def loss_fn(model, target, key=None):
    args = (key,) if key is not None else ()
    pred = model.get_value(*args)
    return (pred - target)**2

grad_fn = eqx.filter_grad(loss_fn)

TRAINABLE_FIXTURES = [
    "param", "log_param", "tanh_param", "gaussian_param", "log_gaussian_param"
]

@pytest.mark.parametrize("model_fixture", TRAINABLE_FIXTURES)
def test_trainable_parameters_have_grads(model_fixture, request, rng_key):
    model = request.getfixturevalue(model_fixture)
    key_arg = rng_key if "gaussian" in model_fixture else None
    
    grads = grad_fn(model, 10.0, key=key_arg)
    
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0, f"No gradients found for {model.__class__.__name__}"
    assert all(leaf is not None for leaf in leaves)

def test_static_parameter_has_no_grad(static_param):
    grads = grad_fn(static_param, 10.0)
    assert grads.value is None
    assert len(jax.tree_util.tree_leaves(grads)) == 0


ALL_FIXTURES = TRAINABLE_FIXTURES + ["static_param"]

@pytest.mark.parametrize("model_fixture", ALL_FIXTURES)
def test_parameter_is_jittable(model_fixture, request, rng_key):
    model = request.getfixturevalue(model_fixture)
    key_arg = rng_key if "gaussian" in model_fixture else None

    # We need a small wrapper for jitting functions with optional args
    def get_val_wrapper(m):
        return m.get_value(key_arg) if key_arg is not None else m.get_value()

    jitted_fn = jax.jit(get_val_wrapper)
    
    try:
        result = jitted_fn(model)
        assert isinstance(result, jax.Array)
    except Exception as e:
        pytest.fail(f"JIT failed for {model.__class__.__name__}: {e}")