# PPO Agent for Kinitro MetaWorld MT10 Challenge
## Installation

### Prerequisites
- Python 3.12+
- MuJoCo (for MetaWorld environments)

### Quick Start

```bash
# Install with uv (recommended)
uv venv --python python 3.12
source .venv/bin/activate
uv sync
```

### Training a Model
```bash
python train --seed 42
```


#### Local Evaluation
```bash
# Evaluate on reach task
python evaluation.py --task <task_name> --episodes <num_episodes> --render-mode <human|rgb_array> --model-path <path_to_trained_model>
```
Available tasks:
```
reach-v3
push-v3
pick-place-v3
door-open-v3
door-close-v3
drawer-open-v3
button-press-topdown-v3
button-press-v3
peg-insert-side-v3
lever-pull-v3
```

## PPO (Proximal Policy Optimization) Details
- **Configuration**: Multi-task learning on MT50
- **Features**: GAE, value function clipping, KL divergence constraint
- **Network**: Continuous action policy with vanilla MLP architecture
- **Training**: 16 epochs, 32 gradient steps per update


### Key Parameters

```python
# PPO Configuration
ppo_config = PPOConfig(
    num_tasks=50,           # Multi-task learning
    gamma=0.99,             # Discount factor
    gae_lambda=0.97,        # GAE parameter
    num_epochs=16,          # Training epochs
    num_gradient_steps=32,  # Gradient steps per update
    target_kl=None,         # KL divergence constraint
)
```

### TensorBoard Integration
```bash
# Training automatically starts TensorBoard
python train.py

# View logs at http://localhost:6006
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MetaWorld](https://github.com/rlworkgroup/metaworld) - Robotics benchmark suite
- [Metaworld Algorithms](https://github.com/rainx0r/metaworld-algorithms) - Base algorithm implementations
- [Kinitro](https://github.com/threetau/kinitro) - Agent submission framework
- [JAX](https://github.com/google/jax) - High-performance machine learning
- [Flax](https://github.com/google/flax) - Neural network library
