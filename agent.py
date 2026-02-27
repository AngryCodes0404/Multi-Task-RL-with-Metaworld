import logging
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from agent_interface import AgentInterface
from checkpoint import get_checkpoint_restore_args
from envs.metaworld import MetaworldConfig
from config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OnPolicyTrainingConfig
from rl.algorithms import PPO, PPOConfig
import orbax.checkpoint as ocp


class RLAgent(AgentInterface):
    """
    PPO-based agent for MetaWorld tasks.

    This agent uses the PPO algorithm from your library and can load trained models
    from checkpoints. It automatically detects the latest checkpoint in ./checkpoints
    if no model path is provided.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        seed: Optional[int] = None,
        max_episode_steps: int = 200,
        model_path: Optional[str] = None,
        num_tasks: int = 50,
        task_name: Optional[str] = None,
    ):

        super().__init__(observation_space, action_space, seed)

        self.max_episode_steps = max_episode_steps
        self.model_path = model_path
        self.num_tasks = num_tasks
        self.task_name_override = task_name
        self.logger = logging.getLogger(__name__)

        # Initialize PPO agent
        self.ppo_agent = None
        self.expected_obs_dim: Optional[int] = None
        self._obs_pad_warning_emitted = False
        self.task_to_index: dict[str, int] = {}
        self.task_one_hot_dim: Optional[int] = None
        self._task_mapping_initialized = False
        self._initialize_ppo_agent()

    def _initialize_ppo_agent(self):
        """Initialize the PPO agent with the configuration."""
        try:
            # Create the environment configuration
            env_config = MetaworldConfig(
                env_id="MT10",
                terminate_on_success=False,
            )

            # Create the PPO configuration
            ppo_config = PPOConfig(
                num_tasks=self.num_tasks,
                gamma=0.99,
                policy_config=ContinuousActionPolicyConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0),
                    )
                ),
                vf_config=ValueFunctionConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0),
                    )
                ),
                num_epochs=16,
                num_gradient_steps=32,
                gae_lambda=0.97,
                target_kl=None,
                clip_vf_loss=False,
            )

            # Initialize the PPO agent
            self.ppo_agent = PPO.initialize(ppo_config, env_config, seed=self.seed)
            obs_space = env_config.observation_space
            if hasattr(obs_space, "shape") and obs_space.shape is not None:
                self.expected_obs_dim = int(np.prod(obs_space.shape))
            else:
                self.expected_obs_dim = None
            if self.task_name_override:
                self._initialize_task_mapping()
            self.logger.info("PPO agent initialized successfully")

            # Load trained model
            self._load_trained_model()

        except Exception as e:
            self.logger.error(f"Failed to initialize PPO agent: {e}", exc_info=True)
            raise

    def _load_trained_model(self):
        """Load a trained PPO model from checkpoint."""
        try:
            if self.model_path is None:
                # Try to find the latest checkpoint in ./checkpoints folder
                checkpoints_dir = Path("./checkpoints")
                if checkpoints_dir.exists():
                    # Look for PPO run directories
                    run_dirs = [
                        d
                        for d in checkpoints_dir.iterdir()
                        if d.is_dir() and "ppo" in d.name.lower()
                    ]
                    latest_checkpoint = None

                    for run_dir in run_dirs:
                        # Look for checkpoints subdirectory
                        run_checkpoints_dir = run_dir / "checkpoints"
                        if run_checkpoints_dir.exists():
                            # Find all checkpoint step directories
                            step_dirs = [
                                d
                                for d in run_checkpoints_dir.iterdir()
                                if d.is_dir() and d.name.isdigit()
                            ]
                            if step_dirs:
                                # Get the latest step directory
                                latest_step = max(step_dirs, key=lambda x: int(x.name))
                                latest_checkpoint = latest_step
                                break

                    if latest_checkpoint:
                        self.model_path = str(latest_checkpoint.absolute())
                        self.logger.info(
                            f"Detected latest checkpoint: {self.model_path}"
                        )
                    else:
                        self.logger.warning(
                            "No PPO checkpoint directories found in ./checkpoints"
                        )
                        return
                else:
                    self.logger.warning(
                        "No model path provided and ./checkpoints directory not found"
                    )
                    return

            checkpoint_manager = ocp.CheckpointManager(
                Path(self.model_path).parent.absolute(),
                item_names=("agent", "env_states", "rngs", "buffer", "metadata"),
                options=ocp.CheckpointManagerOptions(create=False),
            )

            if checkpoint_manager.latest_step() is not None:
                # Restore the agent from checkpoint
                restore_args = get_checkpoint_restore_args(self.ppo_agent)
                ckpt = checkpoint_manager.restore(
                    checkpoint_manager.latest_step(),
                    args=restore_args,
                )
                self.ppo_agent = ckpt["agent"]
                policy_params = self.ppo_agent.policy.params.get("params", {})
                # Attempt to infer observation dimensionality from parameters if available
                try:
                    torso = policy_params[next(iter(policy_params))]
                    network = torso[next(iter(torso))]
                    mlp = network[next(iter(network))]
                    layer = mlp[next(iter(mlp))]
                    kernel = layer["kernel"]
                    self.expected_obs_dim = int(kernel.shape[0])
                except Exception:
                    # Fall back to existing expectation
                    pass
                self.logger.info(
                    f"Loaded trained PPO model from step {checkpoint_manager.latest_step()}"
                )
            else:
                self.logger.warning("No checkpoint found in the specified directory")

        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}", exc_info=True)
            self.logger.info("Continuing with randomly initialized model")

    def act(
        self, observation: Union[np.ndarray, dict], deterministic: bool = True
    ) -> np.ndarray:
        """
        Generate an action given an observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action selection

        Returns:
            Action as numpy array
        Observation structure expected here:
            - Dict with key "observation.state" containing the trimmed proprioceptive
              state plus task one-hot (or a raw numpy array fallback).
            - Optional image entries ("observation.image", "observation.image2", ...),
              which are ignored by this policy.

        """
        if self.ppo_agent is None:
            raise RuntimeError("PPO agent not initialized")

        # Process observation for PPO
        processed_obs = self._process_observation_for_ppo(observation)

        # Convert to JAX array
        obs_jax = jnp.array(processed_obs)

        # Get action from PPO agent (PPO doesn't have deterministic parameter)
        action_jax = self.ppo_agent.eval_action(obs_jax)

        # Convert back to numpy array
        action_np = np.array(action_jax)

        return action_np

    def _initialize_task_mapping(self) -> None:
        """Build mapping from task names to MT10 one-hot indices."""
        if self._task_mapping_initialized:
            return

        try:
            import metaworld  # type: ignore
        except ImportError:
            self.logger.warning(
                "Unable to import MetaWorld while building task mapping; "
                "falling back to zero-padding for missing one-hot encodings."
            )
            self._task_mapping_initialized = True
            return

        task_names: list[str] = []

        try:
            mt10 = metaworld.MT10()
            task_names = list(mt10.train_classes.keys())
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Failed to infer MT10 task order from MetaWorld (%s); "
                "falling back to default ordering.",
                exc,
            )

        if not task_names:
            # Fallback to a documented ordering
            task_names = [
                "reach-v3",
                "push-v3",
                "pick-place-v3",
                "door-open-v3",
                "drawer-open-v3",
                "button-press-topdown-v3",
                "button-press-v3",
                "door-close-v3",
                "peg-insert-side-v3",
                "lever-pull-v3",
            ]

        self.task_one_hot_dim = len(task_names)

        # Include v2/v3 aliases for robustness
        mapping: dict[str, int] = {}
        for idx, name in enumerate(task_names):
            mapping[name] = idx
            if name.endswith("-v2"):
                mapping[name[:-2] + "v3"] = idx
            if name.endswith("-v3"):
                mapping[name[:-2] + "v2"] = idx

        self.task_to_index = mapping
        self._task_mapping_initialized = True

    def _get_task_one_hot(self, length: int) -> Optional[np.ndarray]:
        """Return one-hot vector for the current task if possible."""
        if self.task_name_override is None:
            return None

        if not self._task_mapping_initialized:
            self._initialize_task_mapping()

        if not self.task_to_index:
            return None

        task_key = self.task_name_override
        if task_key not in self.task_to_index:
            self.logger.warning(
                "Task '%s' not found in MT10 mapping; falling back to zero padding.",
                task_key,
            )
            return None

        one_hot_dim = self.task_one_hot_dim or length
        if one_hot_dim != length:
            if one_hot_dim is None:
                one_hot_dim = length
            else:
                self.logger.warning(
                    "Mismatch between expected one-hot size (%d) and "
                    "available padding length (%d); falling back to zeros.",
                    one_hot_dim,
                    length,
                )
                return None

        one_hot = np.zeros(length, dtype=np.float32)
        index = self.task_to_index[task_key]
        if index >= length:
            self.logger.warning(
                "Task index %d is out of bounds for one-hot of size %d; "
                "falling back to zeros.",
                index,
                length,
            )
            return None

        one_hot[index] = 1.0
        return one_hot

    def _process_observation_for_ppo(
        self, observation: Union[np.ndarray, dict]
    ) -> np.ndarray:
        """
        Process observation for PPO agent.

        Args:
            observation: Raw observation from environment

        Returns:
            Processed observation as numpy array
        """
        if isinstance(observation, dict):
            # Wrapper exposes trimmed state with task one-hot under "observation.state"
            if "observation.state" in observation:
                obs = observation["observation.state"]
            else:
                # Fallback for environments without wrapped dict observations
                obs = observation.get(
                    "state", observation.get("observation", observation)
                )
                if isinstance(obs, dict):
                    obs = np.concatenate(list(obs.values()))
        else:
            obs = observation

        # Ensure it's a numpy array
        obs = np.array(obs, dtype=np.float32)

        if obs.ndim > 1:
            obs = obs.flatten()

        if self.expected_obs_dim is not None and obs.size != self.expected_obs_dim:
            if obs.size > self.expected_obs_dim:
                obs = obs[: self.expected_obs_dim]
            else:
                pad_width = self.expected_obs_dim - obs.size
                one_hot = self._get_task_one_hot(pad_width)
                if one_hot is not None:
                    obs = np.concatenate([obs, one_hot]).astype(np.float32, copy=False)
                else:
                    if not self._obs_pad_warning_emitted:
                        self.logger.warning(
                            "Observation size (%d) smaller than expected (%d); padding with zeros. "
                            "Set --model-path to a checkpoint trained on single-task observations "
                            "or ensure evaluation environment provides matching features.",
                            obs.size,
                            self.expected_obs_dim,
                        )
                        self._obs_pad_warning_emitted = True
                    obs = np.concatenate(
                        [obs, np.zeros(pad_width, dtype=np.float32)]
                    ).astype(np.float32, copy=False)

        return obs

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the agent for a new episode.

        Args:
            seed: Optional seed for the new episode
        """
        # PPO agent is stateless during evaluation, so no reset needed
        if seed is not None:
            self.seed = seed
