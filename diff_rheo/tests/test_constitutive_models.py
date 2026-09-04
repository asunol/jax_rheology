import pytest
import jax
import equinox as eqx

from diff_rheo.models import AbstractConstitutiveModel
from diff_rheo.parameters import (
    AbstractParameter, Parameter, LogParameter,
    GaussianParameter, LogGaussianParameter
)

class ConcreteTestModel(AbstractConstitutiveModel):
    a: AbstractParameter
    b: AbstractParameter
    c: AbstractParameter

@pytest.fixture
def fixed_value_model():
    return ConcreteTestModel(a=Parameter(1.0),b=Parameter(2.0), c=Parameter(3.0))

@pytest.fixture
def stochastic_model():
    return ConcreteTestModel(a=GaussianParameter(1.0,0.1),b=LogGaussianParameter(1.0,0.1), c=Parameter(3.0))

def test_get_parameter_values(fixed_value_model):
    values = fixed_value_model.parameter_values
    assert values["a"] == 1.0
    assert values["b"] == 2.0
    assert values["c"] == 3.0

def test_get_instance(fixed_value_model,rng_key):
    instance = fixed_value_model.get_instance(rng_key)
    assert eqx.tree_equal(fixed_value_model, instance)

def test_get_instance_types(stochastic_model, rng_key):
    instance = stochastic_model.get_instance(rng_key)

    assert isinstance(instance, ConcreteTestModel)

    assert isinstance(instance.a, Parameter)
    assert isinstance(instance.b, LogParameter)

    assert isinstance(instance.c, Parameter)

def test_reproducibility_with_same_key(stochastic_model, rng_key):
    instance_1 = stochastic_model.get_instance(rng_key)
    instance_2 = stochastic_model.get_instance(rng_key)

    # The equinox objects should be identical
    assert eqx.tree_equal(instance_1, instance_2)

def test_stochasticity_with_different_keys(stochastic_model, rng_key):
    key_1, key_2 = jax.random.split(rng_key)
    instance_1 = stochastic_model.get_instance(key_1)
    instance_2 = stochastic_model.get_instance(key_2)

    assert not eqx.tree_equal(instance_1, instance_2)

def test_gradients_and_jit_after_sampling(stochastic_model, rng_key):
    def loss_fn(model, key):
        total_loss = 0.0
        keys = jax.random.split(key, 5)
        for subkey in keys:
            instance = model.get_instance(subkey)
            total_loss += instance.a.get_value() + instance.b.get_value()
        return total_loss

    grad_fn = eqx.filter_grad(loss_fn)
    grads = grad_fn(stochastic_model, rng_key)

    assert grads.a.mean is not None
    assert grads.a.std is not None
    assert grads.b.mean is not None
    assert grads.b.std is not None
    assert grads.c.value == 0.0


def test_jit_gradients_after_sampling(stochastic_model, rng_key):
    @eqx.filter_jit
    @eqx.filter_grad
    def jitted_grad_fn(model, key):
        total_loss = 0.0
        keys = jax.random.split(key, 5)
        for subkey in keys:
            instance = model.get_instance(subkey)
            total_loss += instance.a.get_value() + instance.b.get_value()
        return total_loss

    try:
        jitted_grad_fn(stochastic_model, rng_key)
    except Exception as e:
        pytest.fail(f"JIT compilation of gradient function failed: {e}")