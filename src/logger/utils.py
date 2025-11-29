import io

import flax.nnx as nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    return img_array


def plot_solution(model: nnx.Module, dataset) -> np.ndarray:
    t_domain, x_domain = dataset.t_domain, dataset.x_domain

    t_grid = jnp.linspace(t_domain[0], t_domain[1], 100)
    x_grid = jnp.linspace(x_domain[0], x_domain[1], 100)
    T, X = jnp.meshgrid(t_grid, x_grid, indexing='ij')

    t_flat = T.reshape(-1, 1)
    x_flat = X.reshape(-1, 1)

    pred = model(t=t_flat, x=x_flat)['u'].reshape(T.shape)
    target = dataset.solution(t=t_flat, x=x_flat)['u'].reshape(T.shape)
    abs_error = jnp.abs(pred - target)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].imshow(
        pred.T,
        aspect='auto',
        origin='lower',
        extent=[t_domain[0], t_domain[1], x_domain[0], x_domain[1]],
        cmap='jet',
    )
    axes[0].set_title('Prediction')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        target.T,
        aspect='auto',
        origin='lower',
        extent=[t_domain[0], t_domain[1], x_domain[0], x_domain[1]],
        cmap='jet',
    )
    axes[1].set_title('Target (Analytical)')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('x')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        abs_error.T,
        aspect='auto',
        origin='lower',
        extent=[t_domain[0], t_domain[1], x_domain[0], x_domain[1]],
        cmap='jet',
    )
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('x')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    return fig_to_array(fig)
