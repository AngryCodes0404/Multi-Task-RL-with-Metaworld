"""Training script for PPO model on MetaWorld tasks."""

import argparse
import logging
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OnPolicyTrainingConfig
from envs.metaworld import MetaworldConfig
from rl.algorithms import PPOConfig
from run import Run
from monitoring.utils import set_tensorboard_writer, close_tensorboard_writer


@dataclass(frozen=True)
class Args:
    seed: int = 1
    data_dir: Path = Path("./checkpoints")
    resume: bool = False
    tensorboard_port: int = 6006
    no_tensorboard: bool = False
    env_id: str = "MT10"
    run_name: str | None = None
    num_tasks: int | None = None
    logs_dir: Path = Path("./logs")


def parse_args() -> Args:
    """Parse command line arguments without external dependencies."""
    parser = argparse.ArgumentParser(description="Train PPO agent on MetaWorld tasks.")
    parser.add_argument(
        "--seed",
        type=int,
        default=Args.seed,
        help="Random seed for training (default: 1)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Args.data_dir,
        help="Directory to store checkpoints and logs (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=Args.resume,
        help="Resume training from the latest checkpoint if available",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=Args.tensorboard_port,
        help="Port for TensorBoard server (default: 6006)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=Args.no_tensorboard,
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=Args.env_id,
        help="MetaWorld benchmark to train on (e.g. MT10, MT50)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=Args.run_name,
        help="Name of the run directory (defaults to `<env-id>_ppo`)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=Args.num_tasks,
        help="Number of MetaWorld tasks to sample per rollout (auto-inferred by default)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Args.logs_dir,
        help="Directory to store TensorBoard logs (default: ./logs)",
    )

    parsed_args = parser.parse_args()
    return Args(
        seed=parsed_args.seed,
        data_dir=parsed_args.data_dir,
        resume=parsed_args.resume,
        tensorboard_port=parsed_args.tensorboard_port,
        no_tensorboard=parsed_args.no_tensorboard,
        env_id=parsed_args.env_id,
        run_name=parsed_args.run_name,
        num_tasks=parsed_args.num_tasks,
        logs_dir=parsed_args.logs_dir,
    )


def infer_num_tasks(env_id: str) -> int:
    """Infer the number of tasks from the MetaWorld environment id."""
    env_id = env_id.upper()
    if env_id.startswith("MT") and env_id[2:].isdigit():
        return int(env_id[2:])
    raise ValueError(
        f"Unable to infer the number of tasks for env_id='{env_id}'. "
        "Please provide --num-tasks explicitly."
    )


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def launch_tensorboard(log_dir, port=6006):
    """Launch TensorBoard in a separate thread."""

    def run_tensorboard():
        try:
            # Wait a moment for initial logs to be written
            time.sleep(2)

            # Launch TensorBoard
            result = subprocess.run(
                [
                    "tensorboard",
                    "--logdir",
                    str(log_dir),
                    "--port",
                    str(port),
                    "--host",
                    "localhost",
                    "--reload_interval",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except subprocess.CalledProcessError as exc:
            print("Failed to start TensorBoard automatically.")
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr)
            print(
                f"Start it manually with:\n  tensorboard --logdir {log_dir} --port {port}"
            )
        except FileNotFoundError:
            # TensorBoard not installed
            pass

    # Start TensorBoard in background thread
    tb_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tb_thread.start()

    # Give TensorBoard a moment to start
    time.sleep(3)

    # Try to open browser
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        # Browser opening failed, but that's okay
        pass

    return f"http://localhost:{port}"


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger("ppo:train")

    data_dir = args.data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting PPO training with seed {args.seed}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"MetaWorld environment: {args.env_id}")

    env_config = MetaworldConfig(env_id=args.env_id)
    num_tasks = (
        args.num_tasks if args.num_tasks is not None else infer_num_tasks(args.env_id)
    )
    run_name = args.run_name or f"{args.env_id.lower()}_ppo"
    logger.info(f"Run name: {run_name}")
    logger.info(f"Number of tasks: {num_tasks}")

    # Determine TensorBoard usage
    use_tensorboard = not args.no_tensorboard

    # Run-specific directory
    run_dir = data_dir / f"{run_name}_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs_root = args.logs_dir.expanduser()
    logs_root.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard log directory
    tensorboard_log_dir = None
    if use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log_dir = logs_root / f"{run_name}_{args.seed}" / timestamp
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")

    # Setup TensorBoard logging if enabled
    if use_tensorboard and tensorboard_log_dir:
        set_tensorboard_writer(str(tensorboard_log_dir))
        print(
            f"TensorBoard writer initialized for training logging: {tensorboard_log_dir}"
        )
        print(
            f"To view training metrics later, run: tensorboard --logdir {tensorboard_log_dir.parent}"
        )
        logger.info("TensorBoard writer initialized for training logging")

    run = Run(
        run_name=run_name,
        seed=args.seed,
        data_dir=data_dir,
        env=env_config,
        algorithm=PPOConfig(
            num_tasks=num_tasks,
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
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(1e8),
            rollout_steps=2_000,  # Reduced from 10,000 to 2,000 for more frequent logging
            evaluation_frequency=50,  # Evaluate every 50 episodes for more frequent logging
        ),
        checkpoint=True,
        resume=args.resume,
    )

    tensorboard_url = None
    if use_tensorboard and tensorboard_log_dir:
        print("Starting TensorBoard...")
        try:
            tensorboard_url = launch_tensorboard(
                tensorboard_log_dir, args.tensorboard_port
            )
            print(f"TensorBoard available at: {tensorboard_url}")
            print("TensorBoard will show training metrics in real-time")
        except Exception as e:
            print(f"Failed to start TensorBoard: {e}")
            print("Continuing training without TensorBoard...")

    logger.info("Starting training...")
    run.start()
    logger.info("Training completed!")
    if use_tensorboard:
        close_tensorboard_writer()
        logger.info("TensorBoard writer closed")


if __name__ == "__main__":
    main()
