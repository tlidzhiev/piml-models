from typing import Literal

import jax
import jax.numpy as jnp

from src.dataset.heat1d.data import (
    analytical_solution,
    boundary_condition_left,
    boundary_condition_right,
    initial_condition,
)
from src.dataset.pinn_base import PINNDataset


class PINNHeatEquation1DDataset(PINNDataset):
    def __init__(
        self,
        t_domain: tuple[float, float] = (0.0, 1.0),
        x_domain: tuple[float, float] = (0.0, 1.0),
        alpha: float = 0.01,
        ic_batch_size: int = 100,
        bc_batch_size: int = 100,
        collocation_batch_size: tuple[int, int] | int = (100, 100),
        ic_sampling: Literal['random', 'grid'] = 'grid',
        bc_sampling: Literal['random', 'grid'] = 'grid',
        collocation_sampling: Literal['random', 'grid'] = 'grid',
        resample_step: int | None = None,
        seed: int = 42,
    ):
        super().__init__(
            t_domain=t_domain,
            x_domain=x_domain,
            ic_batch_size=ic_batch_size,
            bc_batch_size=bc_batch_size,
            collocation_batch_size=collocation_batch_size,
            ic_sampling=ic_sampling,
            bc_sampling=bc_sampling,
            collocation_sampling=collocation_sampling,
            pde_params={'alpha': alpha},
            resample_step=resample_step,
            seed=seed,
        )

    def solution(self, t: jax.Array, x: jax.Array) -> dict[str, jax.Array]:
        return {'u': analytical_solution(t, x, self.pde_params['alpha'])}

    def _sample_initial_conditions(self) -> dict[str, jax.Array]:
        nx = self.ic_batch_size

        if self.ic_sampling == 'grid':
            x = jnp.linspace(self.x_domain[0], self.x_domain[1], nx)
        else:
            self.rng, subkey = jax.random.split(self.rng)
            x = jax.random.uniform(subkey, (nx,), minval=self.x_domain[0], maxval=self.x_domain[1])

        x = x.reshape(-1, 1)
        t = jnp.full((nx, 1), self.t_domain[0])
        u = initial_condition(x)
        return {'t': t, 'x': x, 'u': u}

    def _sample_boundary_conditions(self) -> list[dict[str, jax.Array]]:
        nt = self.bc_batch_size

        if self.bc_sampling == 'grid':
            t_left = jnp.linspace(self.t_domain[0], self.t_domain[1], nt)
            t_right = jnp.linspace(self.t_domain[0], self.t_domain[1], nt)
        else:
            self.rng, subkey1, subkey2 = jax.random.split(self.rng, 3)
            t_left = jax.random.uniform(
                subkey1, (nt,), minval=self.t_domain[0], maxval=self.t_domain[1]
            )
            t_right = jax.random.uniform(
                subkey2, (nt,), minval=self.t_domain[0], maxval=self.t_domain[1]
            )

        t_left = t_left.reshape(-1, 1)
        t_right = t_right.reshape(-1, 1)

        x_left = jnp.full((nt, 1), self.x_domain[0])
        u_left = boundary_condition_left(t_left)

        x_right = jnp.full((nt, 1), self.x_domain[1])
        u_right = boundary_condition_right(t_right)

        return [
            {'t': t_left, 'x': x_left, 'u': u_left},
            {'t': t_right, 'x': x_right, 'u': u_right},
        ]
