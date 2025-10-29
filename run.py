"""Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/infra/experiment.py"""

import gc
import pathlib
import random
import time
from dataclasses import dataclass

import jax
import numpy as np
import orbax.checkpoint as ocp
# wandb import removed - using TensorBoard only

from checkpoint import (
    Checkpoint,
    get_checkpoint_restore_args,
    get_last_agent_checkpoint_save_args,
    get_metadata_only_restore_args,
    load_env_checkpoints,
)
from config.envs import EnvConfig, MetaLearningEnvConfig
from config.rl import (
    AlgorithmConfig,
    OffPolicyTrainingConfig,
    TrainingConfig,
)
from rl.algorithms import (
    Algorithm,
    OffPolicyAlgorithm,
    get_algorithm_for_config,
)
from rl.algorithms.base import (
    MetaLearningAlgorithm,
    OnPolicyAlgorithm,
)
from metaworld_types import CheckpointMetadata


@dataclass
class Run:
    run_name: str
    seed: int
    data_dir: pathlib.Path

    env: EnvConfig
    algorithm: AlgorithmConfig
    training_config: TrainingConfig

    checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    best_checkpoint_metric: str = "mean_success_rate"
    resume: bool = False

    def __post_init__(self) -> None:
        # wandb functionality removed - using TensorBoard only
        self._timestamp = str(int(time.time()))

    def _get_data_dir(self) -> pathlib.Path:
        return self.data_dir / f"{self.run_name}_{self.seed}"

    def _get_latest_checkpoint_metadata(self) -> CheckpointMetadata | None:
        checkpoint_manager = ocp.CheckpointManager(
            pathlib.Path(self._get_data_dir() / "checkpoints").absolute(),
            item_names=("metadata",),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.max_checkpoints_to_keep,
                create=True,
                best_fn=lambda x: x[self.best_checkpoint_metric],
            ),
        )
        if checkpoint_manager.latest_step() is not None:
            ckpt: Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                checkpoint_manager.latest_step(),
                args=get_metadata_only_restore_args(),
            )
            return ckpt["metadata"]
        else:
            return None

    def enable_wandb(self, **wandb_kwargs) -> None:
        # wandb functionality removed - using TensorBoard only
        pass

        # wandb initialization removed - using TensorBoard only

    def start(self) -> None:
        # Check for available devices more robustly
        try:
            gpu_count = jax.device_count("gpu")
        except RuntimeError:
            gpu_count = 0
        
        try:
            tpu_count = jax.device_count("tpu")
        except RuntimeError:
            tpu_count = 0
            
        if gpu_count < 1 and tpu_count < 1:
            print(f"Warning: No GPU/TPU found, using CPU. Available devices: {jax.devices()}")
            print("Training will be slower on CPU, but will proceed...")

        envs = self.env.spawn(seed=self.seed)

        algorithm_cls = get_algorithm_for_config(self.algorithm)
        algorithm: Algorithm
        algorithm = algorithm_cls.initialize(self.algorithm, self.env, seed=self.seed)
        is_off_policy = isinstance(algorithm, OffPolicyAlgorithm)

        buffer_checkpoint = None
        checkpoint_manager = None
        checkpoint_metadata = None
        envs_checkpoint = None

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.checkpoint:
            checkpoint_items = (
                "agent",
                "env_states",
                "rngs",
                "metadata",
            )
            if is_off_policy:
                checkpoint_items += ("buffer",)

            checkpoint_manager = ocp.CheckpointManager(
                pathlib.Path(self._get_data_dir() / "checkpoints").absolute(),
                item_names=checkpoint_items,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=self.max_checkpoints_to_keep,
                    create=True,
                    best_fn=lambda x: x[self.best_checkpoint_metric],
                ),
            )

            if self.resume and checkpoint_manager.latest_step() is not None:
                if is_off_policy:
                    assert isinstance(self.training_config, OffPolicyTrainingConfig)
                    rb = algorithm.spawn_replay_buffer(
                        self.env,
                        self.training_config,
                    )
                else:
                    rb = None
                ckpt: Checkpoint = checkpoint_manager.restore(  # pyright: ignore [reportAssignmentType]
                    checkpoint_manager.latest_step(),
                    args=get_checkpoint_restore_args(algorithm, rb),
                )
                algorithm = ckpt["agent"]

                if is_off_policy:
                    buffer_checkpoint = ckpt["buffer"]  # pyright: ignore [reportTypedDictNotRequiredAccess]

                envs_checkpoint = ckpt["env_states"]
                load_env_checkpoints(envs, envs_checkpoint)

                random.setstate(ckpt["rngs"]["python_rng_state"])
                np.random.set_state(ckpt["rngs"]["global_numpy_rng_state"])

                checkpoint_metadata: CheckpointMetadata | None = ckpt["metadata"]
                assert checkpoint_metadata is not None

                self._timestamp = checkpoint_metadata.get("timestamp", self._timestamp)

                print(f"Loaded checkpoint at step {checkpoint_metadata['step']}")

        # Track number of params
        # wandb config update removed - using TensorBoard only

        # Train
        agent = algorithm.train(
            config=self.training_config,
            envs=envs,
            env_config=self.env,
            run_timestamp=self._timestamp,
            seed=self.seed,
            track=True,  # Enable TensorBoard logging
            checkpoint_manager=checkpoint_manager,
            checkpoint_metadata=checkpoint_metadata,
            buffer_checkpoint=buffer_checkpoint,
        )

        # Cleanup
        if self.checkpoint:
            if isinstance(
                agent, (OnPolicyAlgorithm, OffPolicyAlgorithm)
            ) and not isinstance(self.env, MetaLearningEnvConfig):
                mean_success_rate, mean_returns, mean_success_per_task = (
                    self.env.evaluate(envs, agent)
                )

            envs.close()
            del envs

            if isinstance(agent, MetaLearningAlgorithm) and isinstance(
                self.env, MetaLearningEnvConfig
            ):
                gc.collect()
                eval_envs = self.env.spawn_test(self.seed)
                mean_success_rate, mean_returns, mean_success_per_task = (
                    self.env.evaluate_metalearning(eval_envs, agent.wrap())
                )
                eval_envs.close()
                del eval_envs
            else:
                raise ValueError("Invalid agent / env combination.")
            final_metrics = {
                "charts/mean_success_rate": float(mean_success_rate),
                "charts/mean_evaluation_return": float(mean_returns),
            } | {
                f"charts/{task_name}_success_rate": float(success_rate)
                for task_name, success_rate in mean_success_per_task.items()
            }
            assert checkpoint_manager is not None
            checkpoint_manager.wait_until_finished()

            if checkpoint_manager._options.max_to_keep is not None:
                checkpoint_manager._options.max_to_keep += 1
            checkpoint_manager.save(
                self.training_config.total_steps + 1,
                args=get_last_agent_checkpoint_save_args(agent, final_metrics),
                metrics={
                    k.removeprefix("charts/"): v for k, v in final_metrics.items()
                },
            )
            checkpoint_manager.wait_until_finished()

            # Log final model checkpoint
            # wandb logging removed - using TensorBoard only

            # wandb best checkpoint logging removed - using TensorBoard only

            checkpoint_manager.close()

        # wandb finish removed - using TensorBoard only
