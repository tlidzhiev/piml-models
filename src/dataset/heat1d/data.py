import jax
import jax.numpy as jnp


def analytical_solution(t: jax.Array, x: jax.Array, alpha: float) -> jax.Array:
    return jnp.exp(-alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)


def initial_condition(x: jax.Array) -> jax.Array:
    return jnp.sin(jnp.pi * x)


def boundary_condition_left(t: jax.Array) -> jax.Array:
    return jnp.zeros_like(t)


def boundary_condition_right(t: jax.Array) -> jax.Array:
    return jnp.zeros_like(t)
