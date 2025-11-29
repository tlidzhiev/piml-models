from typing import Callable, Literal

import flax.nnx as nnx
import jax
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig


def _parse_act(activation: str) -> tuple[str, float | None]:
    """
    Parse activation string to extract name and optional parameter.

    Parameters
    ----------
    activation : str
        Activation string, optionally with parameter (e.g., "leaky_relu:0.2").

    Returns
    -------
    act_name : str
        Activation name in lowercase.
    param : float or None
        Activation parameter if provided, None otherwise.

    Raises
    ------
    ValueError
        If parameter string cannot be converted to float.
    """
    if ':' in activation:
        act_name, param_str = activation.split(':', 1)
        try:
            param = float(param_str)
        except ValueError:
            raise ValueError(
                f'Invalid activation parameter "{param_str}" in "{activation}". '
                f'Parameter must be a number.'
            )
        return act_name.lower(), param
    return activation.lower(), None


def get_activation(activation: str) -> Callable[[jax.Array], jax.Array]:
    """
    Get activation function by name.

    Parameters
    ----------
    activation : str
        Activation name, optionally with parameter (e.g., "leaky_relu:0.2").
        Supported: "relu", "leaky_relu", "tanh", "gelu", "silu".

    Returns
    -------
    Callable
        JAX activation function.

    Raises
    ------
    ValueError
        If activation type is not supported.
    """
    act_name, param = _parse_act(activation)
    match act_name:
        case 'relu':
            return nnx.relu
        case 'leaky_relu':
            slope = param if param is not None else 0.01
            return lambda x: nnx.leaky_relu(x, negative_slope=slope)
        case 'tanh':
            return nnx.tanh
        case 'gelu':
            return nnx.gelu
        case 'silu':
            return nnx.silu
        case _:
            raise ValueError(
                f'Unknown activation type: "{act_name}". '
                f'Supported types: "relu", "leaky_relu", "tanh", "gelu", "silu"'
            )


def get_kernel_initializer(
    activation: str,
    mode: Literal['normal', 'uniform'],
) -> nnx.initializers.Initializer:
    if mode not in ['normal', 'uniform']:
        raise ValueError(
            f'Unknown initialization mode: "{mode}". Supported modes: "normal", "uniform".'
        )

    act_name, param = _parse_act(activation)
    param = param if param is not None else 0.0

    if act_name not in ['relu', 'leaky_relu', 'tanh', 'silu', 'gelu']:
        raise ValueError(
            f"Unknown activation type for initialization: '{act_name}'. "
            f'Supported types: {", ".join(["relu", "leaky_relu", "tanh", "silu", "gelu"])}'
        )
    if act_name == 'tanh':
        return (
            nnx.initializers.xavier_normal()
            if mode == 'normal'
            else nnx.initializers.xavier_uniform()
        )
    else:
        return nnx.initializers.he_normal() if mode == 'normal' else nnx.initializers.he_uniform()


def compute_grad_norm(grads: nnx.State) -> jax.Array:
    grad_norm = jax.tree_util.tree_map(lambda x: jax.numpy.linalg.norm(x, ord=2), grads)
    grad_norm = jax.numpy.sqrt(sum(jax.tree_util.tree_leaves(grad_norm)))
    return grad_norm


def get_optimizer(cfg: DictConfig, model: nnx.Module) -> nnx.Optimizer:
    transformations = []

    if cfg.training.get('max_grad_norm') is not None:
        value = cfg.training['max_grad_norm']
        transformations.append(optax.clip_by_global_norm(value))

    lr_scheduler = instantiate(cfg.lr_scheduler)
    optimizer = instantiate(cfg.optimizer, learning_rate=lr_scheduler)
    transformations.append(optimizer)
    transformations = optax.chain(*transformations)
    return nnx.Optimizer(model, transformations, wrt=nnx.Param)
