import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import metaworld
import numpy as np
from agent import RLAgent  # PPO agent

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


class AgentEvaluator:
    """
    Evaluator for running and assessing the agent in MetaWorld environments.
    Includes TensorBoard logging for performance monitoring.
    """

    def __init__(
        self,
        task_name: str = "reach-v3",
        render_mode: str = "human",
        max_episodes: int = 5,
        max_steps_per_episode: int = 200,
        seed: Optional[int] = None,
        use_tensorboard: bool = True,
        log_dir: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            task_name: Name of the MetaWorld task to run
            render_mode: Rendering mode ("human" for GUI, "rgb_array" for headless)
            max_episodes: Maximum number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            seed: Random seed for reproducibility
            use_tensorboard: Whether to enable TensorBoard logging
            log_dir: Directory for TensorBoard logs (auto-generated if None)
            model_path: Optional path to a trained model checkpoint directory
        """
        self.task_name = task_name
        self.render_mode = render_mode
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.seed = seed or np.random.randint(0, 1000000)
        self.use_tensorboard = use_tensorboard

        self.logger = logging.getLogger(__name__)
        if model_path is not None:
            self.model_path = str(Path(model_path).expanduser())
        else:
            self.model_path = None
        self.env = None
        self.agent = None

        # Statistics tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.success_rate = 0.0

        # TensorBoard setup
        self.tb_writer = None
        self.log_dir = Path(log_dir).expanduser() if log_dir is not None else None
        if self.use_tensorboard:
            if SummaryWriter is None:
                self.logger.warning(
                    "TensorBoard logging requested but torch is not installed; "
                    "disabling TensorBoard logging."
                )
                self.use_tensorboard = False
            else:
                if self.log_dir is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.log_dir = Path("runs") / f"{self.task_name}_{timestamp}"

                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
                self.logger.info(f"TensorBoard logging enabled: {self.log_dir}")
                self.logger.info(
                    f"View logs with: tensorboard --logdir {self.log_dir.parent}"
                )

    def setup_environment(self) -> gym.Env:
        """
        Set up the MetaWorld environment with MuJoCo rendering.

        Returns:
            Configured gymnasium environment
        """
        try:
            # Create MetaWorld environment
            if self.task_name == "reach-v3":
                # Use the reach task that matches our agent's policy
                mt1 = metaworld.MT1(self.task_name, seed=self.seed)
                env = mt1.train_classes[self.task_name]()
                task = mt1.train_tasks[0]
                env.set_task(task)
            else:
                # For other tasks, try to create them directly
                mt1 = metaworld.MT1(self.task_name, seed=self.seed)
                env = mt1.train_classes[self.task_name]()
                task = mt1.train_tasks[0]
                env.set_task(task)

            # Wrap with gymnasium if needed
            if not isinstance(env, gym.Env):
                env = gym.make(env.spec.id if hasattr(env, "spec") else self.task_name)

            # Configure rendering
            if hasattr(env, "render_mode"):
                env.render_mode = self.render_mode

            self.logger.info(f"Environment created: {self.task_name}")
            self.logger.info(f"Observation space: {env.observation_space}")
            self.logger.info(f"Action space: {env.action_space}")

            return env

        except Exception as e:
            self.logger.error(f"Failed to create environment {self.task_name}: {e}")
            self.logger.info("Falling back to reach-v3 environment")

            # Fallback to a simple reach environment
            mt1 = metaworld.MT1("reach-v3", seed=self.seed)
            env = mt1.train_classes["reach-v3"]()
            task = mt1.train_tasks[0]
            env.set_task(task)

            return env

    def setup_agent(self, env: gym.Env, model_path: Optional[str] = None) -> RLAgent:
        """
        Set up the agent with the environment's observation and action spaces.

        Args:
            env: The gymnasium environment
            model_path: Path to trained model checkpoint directory

        Returns:
            Configured RLAgent
        """
        agent = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=self.seed,
            max_episode_steps=self.max_steps_per_episode,
            model_path=model_path,
            task_name=self.task_name,
        )

        self.logger.info("Agent initialized successfully")
        return agent

    def run_episode(self, episode_num: int) -> Dict[str, float]:
        """
        Run a single episode and return statistics.

        Args:
            episode_num: Episode number for logging

        Returns:
            Dictionary containing episode statistics
        """
        obs, info = self.env.reset(seed=self.seed + episode_num)
        self.agent.reset()

        episode_reward = 0.0
        episode_length = 0
        success = False
        step_rewards = []

        self.logger.info(f"Starting episode {episode_num + 1}")

        for step in range(self.max_steps_per_episode):
            try:
                # Get action from agent
                action_tensor = self.agent.act(obs)

                # Convert to numpy array if needed
                if hasattr(action_tensor, "numpy"):
                    action = action_tensor.numpy()
                elif hasattr(action_tensor, "detach"):
                    action = action_tensor.detach().numpy()
                else:
                    action = np.array(action_tensor)

                # Take step in environment
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Render the environment for human viewing
                if self.render_mode == "human":
                    self.env.render()
                    time.sleep(0.02)  # Small delay to make visualization smoother

                episode_reward += reward
                episode_length += 1
                step_rewards.append(reward)

                # Log to TensorBoard (step-level metrics)
                if self.tb_writer:
                    global_step = episode_num * self.max_steps_per_episode + step
                    self.tb_writer.add_scalar("Step/Reward", reward, global_step)
                    self.tb_writer.add_scalar(
                        "Step/CumulativeReward", episode_reward, global_step
                    )

                # Check for success (MetaWorld specific)
                if hasattr(info, "get") and info.get("success", False):
                    success = True

                # Log progress occasionally
                if step % 50 == 0:
                    self.logger.debug(
                        f"Episode {episode_num + 1}, Step {step}: "
                        f"Reward {reward:.3f}, Total {episode_reward:.3f}"
                    )

                if terminated or truncated:
                    break

            except Exception as e:
                self.logger.error(f"Error during step {step}: {e}")
                break

        # Log episode-level metrics to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Episode/Reward", episode_reward, episode_num)
            self.tb_writer.add_scalar("Episode/Length", episode_length, episode_num)
            self.tb_writer.add_scalar("Episode/Success", float(success), episode_num)
            if step_rewards:
                self.tb_writer.add_scalar(
                    "Episode/AvgStepReward", np.mean(step_rewards), episode_num
                )
                self.tb_writer.add_scalar(
                    "Episode/MaxStepReward", np.max(step_rewards), episode_num
                )
                self.tb_writer.add_scalar(
                    "Episode/MinStepReward", np.min(step_rewards), episode_num
                )

        episode_stats = {
            "reward": episode_reward,
            "length": episode_length,
            "success": success,
        }

        self.logger.info(
            f"Episode {episode_num + 1} completed: "
            f"Reward {episode_reward:.3f}, "
            f"Length {episode_length}, "
            f"Success {success}"
        )

        return episode_stats

    def run_evaluation(self):
        """
        Run the complete evaluation session.
        """
        self.logger.info("Starting agent evaluation")

        # Setup environment and agent
        self.env = self.setup_environment()
        self.agent = self.setup_agent(self.env, self.model_path)

        # Run episodes
        total_successes = 0

        for episode in range(self.max_episodes):
            episode_stats = self.run_episode(episode)

            self.episode_rewards.append(episode_stats["reward"])
            self.episode_lengths.append(episode_stats["length"])

            if episode_stats["success"]:
                total_successes += 1

        # Calculate final statistics
        if self.max_episodes > 0:
            self.success_rate = total_successes / self.max_episodes
        else:
            self.success_rate = 0.0

        if self.episode_rewards:
            avg_reward = float(np.mean(self.episode_rewards))
            std_reward = float(np.std(self.episode_rewards))
        else:
            avg_reward = 0.0
            std_reward = 0.0

        if self.episode_lengths:
            avg_length = float(np.mean(self.episode_lengths))
            std_length = float(np.std(self.episode_lengths))
        else:
            avg_length = 0.0
            std_length = 0.0

        # Log summary metrics to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Summary/AvgReward", avg_reward, 0)
            self.tb_writer.add_scalar("Summary/StdReward", std_reward, 0)
            self.tb_writer.add_scalar("Summary/AvgLength", avg_length, 0)
            self.tb_writer.add_scalar("Summary/StdLength", std_length, 0)
            self.tb_writer.add_scalar("Summary/SuccessRate", self.success_rate, 0)

            # Add histogram of rewards and lengths
            if self.episode_rewards:
                self.tb_writer.add_histogram(
                    "Summary/RewardDistribution", np.array(self.episode_rewards), 0
                )
            if self.episode_lengths:
                self.tb_writer.add_histogram(
                    "Summary/LengthDistribution", np.array(self.episode_lengths), 0
                )

            # Add hyperparameters
            self.tb_writer.add_hparams(
                {
                    "task": self.task_name,
                    "episodes": self.max_episodes,
                    "max_steps": self.max_steps_per_episode,
                    "seed": self.seed,
                    "render_mode": self.render_mode,
                },
                {
                    "avg_reward": avg_reward,
                    "success_rate": self.success_rate,
                    "avg_length": avg_length,
                },
            )

            if self.log_dir:
                self.logger.info(
                    "TensorBoard logs saved. View with: "
                    f"tensorboard --logdir {self.log_dir.parent}"
                )

            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

        self.logger.info("=" * 50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Task: {self.task_name}")
        self.logger.info(f"Episodes: {self.max_episodes}")
        self.logger.info(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        self.logger.info(f"Average Length: {avg_length:.1f} ± {std_length:.1f}")
        self.logger.info(f"Success Rate: {self.success_rate:.1%}")
        self.logger.info("=" * 50)

        # Close environment
        if self.env:
            self.env.close()

        return {
            "task": self.task_name,
            "episodes": self.max_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_length": avg_length,
            "std_length": std_length,
            "success_rate": self.success_rate,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }

    def list_available_tasks(self):
        """
        List all available MetaWorld tasks.
        """
        try:
            # Get all MT1 tasks
            mt1_tasks = metaworld.MT1.get_train_tasks()
            self.logger.info("Available MetaWorld MT1 tasks:")
            for i, task in enumerate(mt1_tasks, 1):
                self.logger.info(f"  {i}. {task}")

            # Get all MT10 tasks
            mt10 = metaworld.MT10()
            self.logger.info("\nAvailable MetaWorld MT10 tasks:")
            for i, task in enumerate(mt10.train_classes.keys(), 1):
                self.logger.info(f"  {i}. {task}")

        except Exception as e:
            self.logger.error(f"Error listing tasks: {e}")
            self.logger.info("Some common MetaWorld tasks:")
            common_tasks = [
                "reach-v3",
                "push-v3",
                "pick-place-v3",
                "door-open-v3",
                "drawer-open-v3",
                "button-press-topdown-v3",
                "peg-insert-side-v3",
            ]
            for i, task in enumerate(common_tasks, 1):
                self.logger.info(f"  {i}. {task}")


def setup_logging(level=logging.INFO):
    """Configure logging for the evaluator."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@dataclass(frozen=True)
class Args:
    task: str = "reach-v3"
    episodes: int = 5
    steps: int = 200
    seed: Optional[int] = None
    render_mode: str = "human"
    log_level: str = "INFO"
    list_tasks: bool = False
    log_dir: Optional[Path] = None
    model_path: Optional[Path] = None
    no_tensorboard: bool = False


def parse_args() -> Args:
    """Parse command line arguments for the evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate the MetaWorld agent in MuJoCo"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=Args.task,
        help="MetaWorld task name (default: reach-v3)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=Args.episodes,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=Args.steps,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Args.seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=Args.render_mode,
        choices=["human", "rgb_array"],
        help="Rendering mode (default: human)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=Args.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Args.log_dir,
        help="Directory for TensorBoard logs (defaults to runs/<task>_<timestamp>)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Args.model_path,
        help="Path to a trained model checkpoint directory",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=Args.no_tensorboard,
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        default=Args.list_tasks,
        help="List available MetaWorld tasks and exit",
    )

    parsed = parser.parse_args()
    return Args(
        task=parsed.task,
        episodes=parsed.episodes,
        steps=parsed.steps,
        seed=parsed.seed,
        render_mode=parsed.render_mode,
        log_level=parsed.log_level,
        list_tasks=parsed.list_tasks,
        log_dir=parsed.log_dir,
        model_path=parsed.model_path,
        no_tensorboard=parsed.no_tensorboard,
    )


def main():
    """Main entry point for the evaluator."""
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    evaluator_kwargs = dict(
        task_name=args.task,
        render_mode=args.render_mode,
        max_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        seed=args.seed,
        use_tensorboard=not args.no_tensorboard,
        log_dir=args.log_dir,
        model_path=args.model_path,
    )

    if args.list_tasks:
        AgentEvaluator(**{**evaluator_kwargs, "use_tensorboard": False}).list_available_tasks()
        return

    evaluator = AgentEvaluator(**evaluator_kwargs)

    try:
        results = evaluator.run_evaluation()
    except KeyboardInterrupt:
        logger.info("Evaluation stopped by user")
        return
    except Exception as exc:
        logger.error("Error during evaluation", exc_info=True)
        sys.exit(1)

    # Mirror summary via stdout for quick CLI inspection
    print("\n==== Evaluation Results ====")
    print(f"Task: {results['task']}")
    print(f"Episodes: {results['episodes']}")
    print(f"Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Average Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate: {results['success_rate']:.1%}")


if __name__ == "__main__":
    main()
