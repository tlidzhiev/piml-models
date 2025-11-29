import jax
import jax.numpy as jnp

from src.model.pinn import PINN


def compute_ic_loss(model: PINN, batch: dict[str, jax.Array]) -> jax.Array:
    t, x, u = batch['t'], batch['x'], batch['u']
    u_pred = model(t=t, x=x)['u']
    return jnp.mean((u_pred - u) ** 2)


def compute_bc_loss(model: PINN, batch: list[dict[str, jax.Array]]) -> jax.Array:
    loss_bc = jnp.array(0.0)
    for bc in batch:
        u_pred = model(t=bc['t'], x=bc['x'])['u']
        loss_bc += jnp.mean((u_pred - bc['u']) ** 2)
    return loss_bc


def _compute_derivatives(model: PINN, t, x) -> tuple[jax.Array, jax.Array]:
    def u_fn_t(t_):
        u = model(t=t_.reshape(1, 1), x=x.reshape(1, 1))['u']
        return u.squeeze()

    def u_fn_x(x_):
        u = model(t=t.reshape(1, 1), x=x_.reshape(1, 1))['u']
        return u.squeeze()

    u_t = jax.grad(u_fn_t)(t).reshape(1)
    u_xx = jax.grad(jax.grad(u_fn_x))(x).reshape(1)
    return u_t, u_xx


batch_derivatives = jax.vmap(
    lambda model, t, x: _compute_derivatives(model, t, x),
    in_axes=(None, 0, 0),
)


def compute_pde_loss(
    model: PINN,
    batch: dict[str, jax.Array | dict[str, float]],
) -> jax.Array:
    t, x, alpha = batch['t'], batch['x'], batch['pde_params']['alpha']
    u_t, u_xx = batch_derivatives(model, t.flatten(), x.flatten())  # ty:ignore[possibly-missing-attribute]
    residual = u_t - alpha * u_xx
    loss_pde = jnp.mean(residual**2)
    return loss_pde


def compute_loss(
    model: PINN,
    ic_batch: dict[str, jax.Array],
    bc_batch: list[dict[str, jax.Array]],
    pde_batch: dict[str, jax.Array | dict[str, float]],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    loss_ic = compute_ic_loss(model, ic_batch)
    loss_bc = compute_bc_loss(model, bc_batch)
    loss_pde = compute_pde_loss(model, pde_batch)
    loss = loss_ic + loss_bc + loss_pde
    return loss, {
        'loss': loss,
        'loss_ic': loss_ic,
        'loss_bc': loss_bc,
        'loss_pde': loss_pde,
    }
