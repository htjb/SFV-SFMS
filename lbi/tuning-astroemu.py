"""Train the emulator using astroemu with hyperparameter tuning."""

import os
import sys

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from collections.abc import Generator
from datetime import datetime
from typing import Callable

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from astroemu.network import mlp
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import save as astroemu_save
from astroemu.train import train
from astroemu.utils import compute_mean_std
from optuna import Trial

with open("observations/intrinsic_scatter_per_timescale_errors.yaml") as f:
    rawdata = yaml.safe_load(f)

targets = rawdata.keys()
targets = [
    target for target in targets if not ("M_E" in target or "M_F" in target)
]


def kl_divergence(rmse: np.ndarray) -> float:
    """Calculate the KL divergence based on emulator rmse.

    Based on Bevins et al. 2025.

    Args:
        rmse: the average rmse on the test data.
    """
    meanrmse = np.mean(rmse)
    errors = [
        np.mean([rawdata[t]["high_error"], rawdata[t]["low_error"]])
        for t in targets
    ]
    kl = 6 / 2 * (meanrmse / np.mean(errors)) ** 2
    print(f"DKL Mean: {kl}, Error: {meanrmse}")
    return float(kl)


class InMemoryDataset:
    """In-memory dataset matching the SpectrumDataset interface for astroemu.

    Stores kernel parameters, scatter values, and timescale values in memory
    and provides the get_batch_iterator interface expected by astroemu's
    train(), compute_mean_std(), and save() functions.

    When tiling=True (training), each (params, stds) sample is expanded over
    the timescale axis so the network learns f(log_time, params) -> scatter.
    When tiling=False (statistics), yields (stds, times, params) per batch.

    Args:
        params: Kernel parameters, shape (n_samples, n_params).
        stds: Scatter values in dex, shape (n_samples, n_times).
        times: Timescale values in Gyr, shape (n_times,).
        forward_pipeline: Normalisation pipeline(s) to apply.
        tiling: If True tile samples over timescales (use for training).
    """

    def __init__(
            self, params: jnp.ndarray, stds:jnp.ndarray,
            times: jnp.ndarray, forward_pipeline: Callable | None =None, tiling: bool =True
    ):
        self._params = jnp.array(params, dtype=jnp.float32)
        self._stds = jnp.array(stds, dtype=jnp.float32)
        self._times = jnp.array(times, dtype=jnp.float32)
        self.tiling = tiling
        self.forward_pipeline = (
            forward_pipeline
            if isinstance(forward_pipeline, list)
            else [forward_pipeline]
            if forward_pipeline is not None
            else []
        )
        # Attributes required by astroemu's save()
        self.files = []
        self.x = "times"
        self.y = "stds"
        self.varied_input = None
        self.allow_pickle = False

    def __len__(self) -> int:
        return int(self._params.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        """Return (stds, times, params) for a single sample (un-normalised).

        Matches SpectrumDataset.__getitem__ which returns (y, x, params).
        astroemu's train() calls dataset[0] to infer input dimensionality.
        """
        return self._stds[idx], self._times, self._params[idx]

    def get_batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        key: jax.Array | None = None,
    ) -> Generator:
        """Yield batches matching SpectrumDataset.get_batch_iterator.

        tiling=True  -> yields (specs_flat, concat_inputs) as expected by
                        astroemu's train() and val steps.
        tiling=False -> yields (specs, x, inputs) as expected by
                        astroemu's compute_mean_std().

        Args:
            batch_size: Number of parameter sets per batch.
            shuffle: Whether to shuffle sample order.
            key: JAX PRNGKey for shuffling. If None, defaults to key 0.
        """
        n = len(self)
        indices = jnp.arange(n)
        if shuffle:
            if key is None:
                key = jax.random.PRNGKey(0)
            indices = jax.random.permutation(key, indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            specs, x, inputs = zip(*[self[int(i)] for i in batch_idx])
            specs = jnp.stack(specs)  # (batch, n_times)
            x = jnp.stack(x)  # (batch, n_times)
            inputs = jnp.stack(inputs)  # (batch, n_params)

            for pipeline in self.forward_pipeline:
                specs, x, inputs = pipeline.forward(specs, x, inputs)

            if self.tiling:
                # Tile params to match each timescale point, prepend x column
                inputs = jnp.repeat(inputs, repeats=specs.shape[-1], axis=0)
                inputs = jnp.concatenate(
                    [x.flatten()[:, None], inputs], axis=-1
                )
                specs_flat = specs.flatten()

                # Shuffle tiled samples so network doesn't see x in order
                if shuffle:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, len(specs_flat))
                    specs_flat = specs_flat[perm]
                    inputs = inputs[perm]

                yield specs_flat, inputs
            else:
                yield specs, x, inputs


def make_pipeline_and_datasets(
    train_params: np.ndarray,
    train_stds: np.ndarray,
    val_params: np.ndarray,
    val_stds: np.ndarray,
    test_params: np.ndarray,
    test_stds: np.ndarray,
    times: np.ndarray,
    log_input: bool,
) -> tuple:
    """Build normalisation pipeline and InMemoryDatasets.

    Computes log10 of timescales (always) and optionally of kernel parameters,
    then z-scores everything using training-set statistics via compute_mean_std.

    Args:
        train_params: Raw training kernel parameters.
        train_stds: Raw training scatter values (dex).
        val_params: Raw validation kernel parameters.
        val_stds: Raw validation scatter values (dex).
        test_params: Raw test kernel parameters.
        test_stds: Raw test scatter values (dex).
        times: Timescale values in Gyr, shape (n_times,).
        log_input: Whether to log10-transform the kernel parameters.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, pipeline, std_pipe).
    """
    log_pipe = log_base_10(
        log_all_params=log_input, log_all_x=True, log_all_y=True
    )
    stats_ds = InMemoryDataset(
        train_params,
        train_stds,
        times,
        forward_pipeline=log_pipe,
        tiling=False,
    )
    mean_spec, std_spec, mean_x, std_x, mean_params, std_params = (
        compute_mean_std(stats_ds.get_batch_iterator(64, shuffle=False))
    )
    std_pipe = standardise(
        y_mean=mean_spec,
        y_std=std_spec,
        x_mean=mean_x,
        x_std=std_x,
        params_mean=mean_params,
        params_std=std_params,
        standardise_y=True,
        standardise_x=True,
        standardise_params=True,
    )
    pipeline = [log_pipe, std_pipe]

    train_ds = InMemoryDataset(
        train_params, train_stds, times, forward_pipeline=pipeline, tiling=True
    )
    val_ds = InMemoryDataset(
        val_params, val_stds, times, forward_pipeline=pipeline, tiling=True
    )
    test_ds = InMemoryDataset(
        test_params, test_stds, times, forward_pipeline=pipeline, tiling=True
    )
    return train_ds, val_ds, test_ds, pipeline, std_pipe


def predict(
    best_params: dict,
    test_ds: InMemoryDataset,
    activation: str,
    std_pipe: standardise,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained network on the test set and denormalise outputs.

    Args:
        best_params: Best network parameters from train().
        test_ds: Normalised test dataset (tiling=True).
        activation: Activation function name used during training.
        std_pipe: The standardise pipeline, used to invert z-score on outputs.

    Returns:
        Tuple of (outputs, ground_truth) both shape (n_test, n_times) in dex.
    """
    n_test = len(test_ds)
    n_times = test_ds._times.shape[0]

    all_preds, all_targets = [], []
    for targets, inputs in test_ds.get_batch_iterator(
        batch_size=512, shuffle=False
    ):
        preds = mlp(best_params, inputs, act=activation).squeeze(-1)
        all_preds.append(np.array(preds))
        all_targets.append(np.array(targets))

    preds = np.concatenate(all_preds).reshape(n_test, n_times)
    targets = np.concatenate(all_targets).reshape(n_test, n_times)

    # Undo z-score → log10(dex), then undo log10 → dex
    outputs = 10.0 ** (
        preds * np.array(std_pipe.y_std) + np.array(std_pipe.y_mean)
    )
    ground_truth = 10.0 ** (
        targets * np.array(std_pipe.y_std) + np.array(std_pipe.y_mean)
    )

    return outputs, ground_truth


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
kernel = sys.argv[1]  # "ext-reg" or "sho"
mass_bin = sys.argv[2]  # "8_8.5" or "9_9.5"
zval = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
nparams = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
ngals = int(sys.argv[5]) if len(sys.argv) > 5 else 10000

# Convert mass_bin to format like "8.0_8.5" for consistency with data files
mass_bin = (
    str(float(mass_bin.split("_")[0]))
    + "_"
    + str(float(mass_bin.split("_")[1]))
)


base_dir = (
    "training-data/"
    + kernel
    + "-sfr_theory_mean-"
    + str(nparams)
    + "-"
    + str(ngals)
    + "/"
)
data_file = (
    base_dir
    + "emulation_data_z_"
    + str(zval)
    + "_mass_bin_"
    + mass_bin
    + ".h5"
)
emulator_dir = (
    "lbi/fit_burstiness_project/burstiness-emulator-astroemu-"
    + kernel
    + f"-sfr_theory_mean-{nparams}-{ngals}_z_{zval}_mass_bin_{mass_bin}/"
)
ndims = 3 if kernel == "sho" else 5
os.makedirs(emulator_dir, exist_ok=True)

now = datetime.now()
current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current DateTime:", current_datetime)
np.savetxt(
    os.path.join(emulator_dir, "tuning_time.txt"),
    np.array([current_datetime]),
    fmt="%s",
)

with h5py.File(data_file, "r") as f:
    raw_params = f["parameters"][:]
    raw_stds = f["scatters"][:]
    times = np.array(f.attrs["times"])  # (n_times,) in Gyr

mask = np.all(np.isfinite(raw_params), axis=1) & np.all(
    np.isfinite(raw_stds), axis=1
)
raw_params = raw_params[mask]
raw_stds = raw_stds[mask]

# Train / val / test split
rng = np.random.default_rng(42)
idx = rng.permutation(len(raw_params))
n_test = int(len(idx) * 0.33)
n_val = n_test // 2

train_idx = idx[n_test:]
val_idx = idx[:n_val]
test_idx = idx[n_val:n_test]

train_params, train_stds = raw_params[train_idx], raw_stds[train_idx]
val_params, val_stds = raw_params[val_idx], raw_stds[val_idx]
test_params, test_stds = raw_params[test_idx], raw_stds[test_idx]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: Trial) -> float:
    """Optuna objective: minimise estimated KL divergence.

    Args:
        trial: An optuna trial object.

    Returns:
        Estimated KL divergence between true and emulated posterior.
    """
    nlayers = trial.suggest_int("nlayers", 1, 10)
    hidden_size = trial.suggest_int("hidden_size", ndims + 1, 32)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    activation = trial.suggest_categorical(
        "activation", ["relu", "leaky_relu", "tanh", "gelu"]
    )
    log_input = trial.suggest_categorical("log_input", [True, False])

    train_ds, val_ds, test_ds, pipeline, std_pipe = make_pipeline_and_datasets(
        train_params,
        train_stds,
        val_params,
        val_stds,
        test_params,
        test_stds,
        times,
        log_input,
    )

    best_params, train_losses, val_losses = train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        hidden_size=hidden_size,
        nlayers=nlayers,
        act=activation,
        epochs=250,
        patience=50,
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        key=trial.number,
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(emulator_dir + "training_loss_" + str(trial.number) + ".png")
    plt.close()

    outputs, ground_truth = predict(best_params, test_ds, activation, std_pipe)
    rmse = np.sqrt(np.mean((outputs - ground_truth) ** 2, axis=1))
    return kl_divergence(rmse)


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
try:
    with open(os.path.join(emulator_dir, "best_hyperparameters.yaml")) as f:
        best_hyperparams = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:

    def early_stop_callback(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        """Stop optimization early if no improvement.

        Args:
            study: The Optuna study object.
            trial: The Optuna trial object that just completed.
        """
        n_warmup = 20
        patience = 10
        if len(study.trials) < n_warmup + patience:
            return
        recent_best = min(
            t.value for t in study.trials[-patience:] if t.value is not None
        )
        if recent_best > study.best_value:
            study.stop()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[early_stop_callback])

    print("Best hyperparameters:", study.best_params)
    with open(
        os.path.join(emulator_dir, "best_hyperparameters.yaml"), "w"
    ) as f:
        yaml.dump(study.best_params, f)

    trial_data = pd.DataFrame([t.params for t in study.trials])
    trial_data["loss"] = [t.value for t in study.trials]

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(emulator_dir, "param_importances.png"))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(emulator_dir, "optimization_history.png"))

    best_hyperparams = study.best_params

# ---------------------------------------------------------------------------
# Final training with best hyperparameters
# ---------------------------------------------------------------------------
log_input = best_hyperparams.pop("log_input")
lr = best_hyperparams.pop("learning_rate")
weight_decay = best_hyperparams.pop("weight_decay")
batch_size = best_hyperparams.pop("batch_size")
activation = best_hyperparams.pop("activation")

train_ds, val_ds, test_ds, pipeline, std_pipe = make_pipeline_and_datasets(
    train_params,
    train_stds,
    val_params,
    val_stds,
    test_params,
    test_stds,
    times,
    log_input,
)

best_params, train_losses, val_losses = train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    act=activation,
    learning_rate=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
    epochs=10000,
    patience=200,
    key=0,
    **best_hyperparams,
)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(emulator_dir + "training_loss.png")
plt.close()

# ---------------------------------------------------------------------------
# Save with astroemu's save() — note: load() will recover params but not
# datasets (files=[]), as data lives in memory rather than .npz files.
# ---------------------------------------------------------------------------
astroemu_save(
    os.path.join(emulator_dir, "emulator"),
    best_params,
    train_losses,
    val_losses,
    hidden_size=best_hyperparams["hidden_size"],
    nlayers=best_hyperparams["nlayers"],
    act=activation,
    loss="mse",
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
)

# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------
outputs, ground_truth = predict(best_params, test_ds, activation, std_pipe)

errors = np.abs((outputs - ground_truth) / ground_truth) * 100
mask = np.isfinite(errors)
errors = errors[mask]

print(f"Mean Error:      {errors.mean():.2f}%")
print(f"Median Error:    {np.median(errors):.2f}%")
print(f"Max Error:       {errors.max():.2f}%")
print(f"95th Percentile: {np.percentile(errors, 95):.2f}%")
rmse = np.sqrt(np.mean((outputs - ground_truth) ** 2, axis=1))
print("KL Divergence:  ", kl_divergence(rmse))

np.savetxt(emulator_dir + "mean_error.txt", [errors.mean()])
np.savetxt(emulator_dir + "median_error.txt", [np.median(errors)])
np.savetxt(emulator_dir + "max_error.txt", [errors.max()])
np.savetxt(
    emulator_dir + "95th_percentile_error.txt", [np.percentile(errors, 95)]
)
np.savetxt(emulator_dir + "test_stds.txt", ground_truth)
np.savetxt(emulator_dir + "test_params.txt", test_params)

plt.hist(errors.flatten(), bins=50, label=f"Mean: {errors.mean():.2f}%")
plt.axvline(
    np.median(errors), color="r", label=f"Median: {np.median(errors):.2f}%"
)
plt.axvline(
    np.percentile(errors, 95),
    color="g",
    label=f"95th pct: {np.percentile(errors, 95):.2f}%",
)
plt.legend()
plt.xlabel("Error (%)")
plt.ylabel("Count")
plt.savefig(emulator_dir + "error-distribution.png")
plt.close()

n_test = len(test_ds)
idx_sorted = np.argsort(np.mean(errors.reshape(n_test, -1), axis=1))
gt_sorted = ground_truth[idx_sorted]
out_sorted = outputs[idx_sorted]

fig, axes = plt.subplots(3, 1, figsize=(10, 15))
for ax, gt_row, out_row, title in zip(
    axes,
    [gt_sorted[0], gt_sorted[len(gt_sorted) // 2], gt_sorted[-1]],
    [out_sorted[0], out_sorted[len(out_sorted) // 2], out_sorted[-1]],
    ["Best Prediction", "Median Prediction", "Worst Prediction"],
):
    ax.plot(times, gt_row, label="Ground Truth")
    ax.plot(times, out_row, label="Predicted")
    ax.set_xscale("log")
    ax.set_xlabel("Timescale (Gyr)")
    ax.set_ylabel("Scatter (dex)")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig(emulator_dir + "best-median-worst-prediction.png")
plt.close()
