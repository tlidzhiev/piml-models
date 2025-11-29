from typing import Literal

import jax
import jax.numpy as jnp


class PINNDataset:
    def __init__(
        self,
        t_domain: tuple[float, float],
        x_domain: tuple[float, float],
        ic_batch_size: int = 100,
        bc_batch_size: int = 100,
        collocation_batch_size: tuple[int, int] | int = (100, 100),
        ic_sampling: Literal['random', 'grid'] = 'grid',
        bc_sampling: Literal['random', 'grid'] = 'grid',
        collocation_sampling: Literal['random', 'grid'] = 'grid',
        pde_params: dict[str, float] | None = None,
        resample_step: int | None = None,
        seed: int = 42,
    ):
        if isinstance(collocation_batch_size, int) and collocation_sampling != 'random':
            raise ValueError(
                'When collocation_batch_size is int, collocation_sampling must be "random"'
            )
        # PDE params
        self.t_domain = tuple(t_domain)
        self.x_domain = tuple(x_domain)
        self.pde_params = pde_params if pde_params is not None else {}

        # Data
        self.ic_batch_size = ic_batch_size
        self.bc_batch_size = bc_batch_size
        self.collocation_batch_size = (
            tuple(collocation_batch_size)
            if not isinstance(collocation_batch_size, int)
            else collocation_batch_size
        )
        self.ic_sampling, self.bc_sampling = ic_sampling, bc_sampling
        self.collocation_sampling = collocation_sampling
        self.resample_step = resample_step
        self.ic_data: dict[str, jax.Array] = None
        self.bc_data: list[dict[str, jax.Array]] = None
        self.collocation_data: dict[str, jax.Array | dict[str, float]] = None

        # Current step counter for resampling
        self._current_step = 0

        # Sample initial data
        self.rng = jax.random.PRNGKey(seed)
        self._initialize_data()

    def _initialize_data(self):
        self.ic_data = self._sample_initial_conditions()
        self.bc_data = self._sample_boundary_conditions()
        self.collocation_data = self._sample_collocation_points()
        self._current_step += 1

    def get_batch(
        self,
    ) -> tuple[
        dict[str, jax.Array],
        list[dict[str, jax.Array]],
        dict[str, jax.Array | dict[str, float]],
    ]:
        if self.resample_step is not None and self._current_step % self.resample_step == 0:
            if self.ic_sampling != 'grid':
                self.ic_data = self._sample_initial_conditions()

            if self.bc_sampling != 'grid':
                self.bc_data = self._sample_boundary_conditions()

            if self.collocation_sampling != 'grid':
                self.collocation_data = self._sample_collocation_points()

        self._current_step += 1
        return self.ic_data, self.bc_data, self.collocation_data  # ty: ignore[invalid-return-type]

    def _sample_initial_conditions(self) -> dict[str, jax.Array]:
        raise NotImplementedError(
            f'{type(self).__name__} should implement sample_initial_conditions method.'
        )

    def _sample_boundary_conditions(self) -> list[dict[str, jax.Array]]:
        raise NotImplementedError(
            f'{type(self).__name__} should implement sample_boundary_conditions method.'
        )

    def _sample_random(
        self,
        num_points: tuple[int, int] | int,
        t_range: tuple[float, float],
        x_range: tuple[float, float],
    ) -> dict[str, jax.Array]:
        self.rng, subkey1, subkey2 = jax.random.split(self.rng, 3)

        if isinstance(num_points, int):
            total_points = num_points
            t = jax.random.uniform(subkey2, (total_points, 1), minval=t_range[0], maxval=t_range[1])
            x = jax.random.uniform(subkey1, (total_points, 1), minval=x_range[0], maxval=x_range[1])
        else:
            nt, nx = num_points[0], num_points[1]
            t = jax.random.uniform(subkey2, (nt, 1), minval=t_range[0], maxval=t_range[1])
            x = jax.random.uniform(subkey1, (nx, 1), minval=x_range[0], maxval=x_range[1])
        return {'t': t, 'x': x}

    def _sample_grid(
        self,
        num_points: tuple[int, int],
        t_range: tuple[float, float],
        x_range: tuple[float, float],
    ) -> dict[str, jax.Array]:
        nt, nx = num_points

        t_grid = jnp.linspace(t_range[0], t_range[1], nt)
        x_grid = jnp.linspace(x_range[0], x_range[1], nx)
        T, X = jnp.meshgrid(t_grid, x_grid)

        t = T.flatten().reshape(-1, 1)
        x = X.flatten().reshape(-1, 1)
        return {'t': t, 'x': x}

    def _sample_points(
        self,
        num_points: tuple[int, int] | int,
        sampling_strategy: Literal['random', 'grid'],
        t_range: tuple[float, float],
        x_range: tuple[float, float],
    ) -> dict[str, jax.Array]:
        if sampling_strategy == 'random':
            return self._sample_random(num_points, t_range, x_range)
        elif sampling_strategy == 'grid':
            assert isinstance(num_points, tuple), 'num_points must be a tuple for grid sampling'
            return self._sample_grid(num_points, t_range, x_range)
        else:
            raise ValueError(f'Unknown sampling strategy: {sampling_strategy}')

    def _sample_collocation_points(self) -> dict[str, jax.Array | dict[str, float]]:
        batch = {}
        points = self._sample_points(
            num_points=self.collocation_batch_size,
            sampling_strategy=self.collocation_sampling,
            t_range=self.t_domain,
            x_range=self.x_domain,
        )
        batch.update(points)
        batch.update({'pde_params': self.pde_params})
        return batch
