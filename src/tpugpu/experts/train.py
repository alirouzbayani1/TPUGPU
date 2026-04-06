from __future__ import annotations

import os
from dataclasses import asdict

import flax
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from tpugpu.config import ExpertTrainConfig
from tpugpu.data.mnist import batch_iterator, filter_by_class_ids, load_mnist_numpy
from tpugpu.experts.model import SmallConditionalUNet


class TrainState(train_state.TrainState):
    pass


def make_beta_schedule(num_steps: int) -> jnp.ndarray:
    return jnp.linspace(1e-4, 2e-2, num_steps, dtype=jnp.float32)


def create_train_state(config: ExpertTrainConfig, rng: jax.Array) -> TrainState:
    model = SmallConditionalUNet(
        hidden_channels=config.hidden_channels,
        num_classes=config.num_classes,
        out_channels=config.num_channels,
    )
    dummy_x = jnp.zeros((config.batch_size, config.image_size, config.image_size, config.num_channels))
    dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.float32)
    dummy_y = jnp.zeros((config.batch_size,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t, dummy_y)["params"]
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def ddpm_loss(
    params: flax.core.FrozenDict,
    state: TrainState,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    betas: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    images = batch["images"]
    labels = batch["labels"]
    noise_rng, timestep_rng = jax.random.split(rng)

    noise = jax.random.normal(noise_rng, images.shape)
    timesteps = jax.random.randint(timestep_rng, (images.shape[0],), 0, betas.shape[0])

    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)
    a_t = alpha_cumprod[timesteps]
    a_t = a_t[:, None, None, None]
    noisy_images = jnp.sqrt(a_t) * images + jnp.sqrt(1.0 - a_t) * noise

    t_normalized = timesteps.astype(jnp.float32) / betas.shape[0]
    pred_noise = state.apply_fn({"params": params}, noisy_images, t_normalized, labels)
    loss = jnp.mean((pred_noise - noise) ** 2)
    metrics = {"loss": loss}
    return loss, metrics


@jax.jit
def train_step(
    state: TrainState,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    betas: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    grad_fn = jax.value_and_grad(ddpm_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state, batch, rng, betas)
    state = state.apply_gradients(grads=grads)
    metrics["loss"] = loss
    return state, metrics


def save_checkpoint(state: TrainState, checkpoint_dir: str, step: int, metadata: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = {"state": state, "metadata": metadata}
    path = os.path.join(checkpoint_dir, f"step_{step}")
    checkpointer.save(path, ckpt, force=True)


def train_expert(config: ExpertTrainConfig) -> None:
    train_ds, test_ds = load_mnist_numpy(image_size=config.image_size)
    train_ds = filter_by_class_ids(train_ds, config.class_ids)
    test_ds = filter_by_class_ids(test_ds, config.class_ids)

    rng = jax.random.PRNGKey(config.seed)
    init_rng, loop_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng)
    betas = make_beta_schedule(config.num_diffusion_steps)

    print("config:", asdict(config))
    print("train samples:", len(train_ds.labels))
    print("eval samples:", len(test_ds.labels))
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    global_step = 0
    for epoch in range(config.num_epochs):
        losses = []
        for batch in batch_iterator(train_ds, config.batch_size, seed=config.seed + epoch):
            loop_rng, step_rng = jax.random.split(loop_rng)
            state, metrics = train_step(state, batch, step_rng, betas)
            loss = float(metrics["loss"])
            losses.append(loss)
            global_step += 1
            if global_step % config.log_every_steps == 0:
                print(f"epoch={epoch+1} step={global_step} loss={loss:.6f}")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch+1} train_loss={mean_loss:.6f}")

    metadata = {"config": asdict(config), "global_step": global_step}
    checkpoint_dir = os.path.join(config.checkpoint_dir, config.expert_name)
    save_checkpoint(state, checkpoint_dir, global_step, metadata)
    print(f"saved checkpoint to {checkpoint_dir}")
