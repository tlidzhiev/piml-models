import math

import flax.nnx as nnx
import hydra
import jax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.logger.utils import plot_solution
from src.loss.heat1d.pinn import compute_loss
from src.model.pinn import PINN
from src.utils.jax import compute_grad_norm, get_optimizer


@nnx.jit
def train_step(
    model: PINN,
    optimizer: nnx.Optimizer,
    ic: dict[str, jax.Array],
    bc: list[dict[str, jax.Array]],
    pde: dict[str, jax.Array | dict[str, float]],
) -> dict[str, jax.Array]:
    def loss_fn(model):
        total_loss = compute_loss(model, ic, bc, pde)
        return total_loss

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    metrics['grad_norm'] = compute_grad_norm(grads)
    optimizer.update(model, grads)
    return metrics


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    train_dataset = instantiate(cfg.dataset)

    model = instantiate(cfg.model)
    optimizer = get_optimizer(cfg, model)

    project_config = OmegaConf.to_container(cfg, resolve=True)
    writer = instantiate(cfg.writer, project_config)

    for step in tqdm(range(1, cfg.training.num_steps + 1), desc='Training...'):
        writer.set_step(step - 1)
        ic, bc, pde = train_dataset.get_batch()
        metrics = train_step(model, optimizer, ic, bc, pde)
        writer.add_scalars({k: math.log10(v.item()) for k, v in metrics.items()})

        if step % cfg.training.save_step == 0:
            fig = plot_solution(model, train_dataset)
            writer.add_image(f'solution-step-{step}', fig)
    writer.finish()


if __name__ == '__main__':
    main()
