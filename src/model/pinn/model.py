from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from src.utils.jax import get_activation, get_kernel_initializer


class PINN(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        output_names: list[str],
        activation: str = 'tanh',
        init_mode: Literal['normal', 'uniform'] = 'normal',
        dtype: Literal['float32', 'float64'] = 'float32',
        rngs: int = 0,
    ):
        if len(output_names) != output_dim:
            raise ValueError(
                f'Length of output_names ({len(output_names)}) must match output_dim ({output_dim})'
            )

        _rngs = nnx.Rngs(rngs)
        activation_fn = get_activation(activation)
        kernel_init = get_kernel_initializer(activation, init_mode)
        bias_init = nnx.initializers.zeros_init()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                nnx.Linear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    rngs=_rngs,
                )
            )
            if i < len(dims) - 2:
                layers.append(activation_fn)
        self.layers = nnx.Sequential(*layers)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_names = output_names
        self.activation_str = activation
        self.init_mode = init_mode
        self.dtype_str = dtype

    def __call__(
        self,
        t: jax.Array,
        x: jax.Array | list[jax.Array],
    ) -> dict[str, jax.Array]:
        if t.ndim == 1:
            t = t.reshape(-1, 1)

        if isinstance(x, list):
            x_arrays = []
            for xi in x:
                if xi.ndim == 1:
                    xi = xi.reshape(-1, 1)
                x_arrays.append(xi)
            x_concat = jnp.concatenate(x_arrays, axis=1)
        else:
            if x.ndim == 1:
                x_concat = x.reshape(-1, 1)
            else:
                x_concat = x

        inputs = jnp.concatenate([t, x_concat], axis=1)
        out = self.layers(inputs)

        if self.output_dim == 1:
            return {self.output_names[0]: out}
        else:
            result = {}
            for i, name in enumerate(self.output_names):
                result[name] = out[:, i : i + 1]
            return result
