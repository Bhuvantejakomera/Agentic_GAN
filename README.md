# GAN Agent Project: Autonomous Agent-Guided GAN Training

A PyTorch-based framework for training GANs and DCGANs with autonomous agents that monitor training behaviour and make adjustments during training.

Instead of relying entirely on fixed hyperparameters, the framework uses feedback from the generator and discriminator losses to adapt learning rates and batch sizes. It also includes agents for latent-space exploration and FID-based early stopping.

## Overview

GAN training can be difficult to stabilize. Problems such as mode collapse, discriminator saturation, or an imbalanced generator/discriminator can require frequent manual tuning.

This project explores a simple agent-based approach to this problem. During training, the agents observe training metrics and make predefined adjustments when certain conditions are detected.

The framework currently supports:

- MNIST with a fully connected GAN
- Fashion-MNIST with a DCGAN
- CelebA with a 64×64 RGB DCGAN
- Dynamic learning-rate and batch-size adjustments
- Latent-space exploration
- FID-based early stopping
- Inception Score and loss tracking
- Training logs and generated sample visualization

## Key Components

### Hyperparameter Agent

`HyperparameterAgent` monitors the relationship between generator and discriminator losses:

\[
Ratio = \frac{L_G}{L_D}
\]

Based on this ratio, the agent can adjust the learning rate and batch size during training.

The goal is to prevent situations where one network becomes significantly stronger than the other.

### Stop Agent

`StopAgent` monitors FID scores during training.

Training can be stopped when:

- The target FID is reached.
- The FID score stops improving for a specified number of epochs.

This avoids continuing training when the model is no longer making meaningful progress.

### Latent Agent

`LatentAgent` adds controlled Gaussian noise to the latent vectors used by the generator.

This provides a simple mechanism for exploring different regions of the latent space while keeping the perturbations bounded.

## System Architecture

```text
                         Training Loop
                              |
                              | Epoch Losses
                              v
        +---------------------+---------------------+
        |                                           |
        v                                           v
   Generator (G)                            Discriminator (D)
        |                                           |
        +------------------+------------------------+
                           |
                           v
                 Loss Ratio Evaluator
                           |
                       L_G / L_D
                           |
                           v
                 Hyperparameter Agent
                           |
                 +---------+---------+
                 |                   |
              Adjust LR          Adjust Batch Size
                 |                   |
                 +---------+---------+
                           |
                           v
                    Training Output
                           |
                           v
                  outputs/logs/agent.log
```

## Agent Decision Logic

| Condition | Loss Ratio `L_G / L_D` | Diagnosis | Action |
|---|---:|---|---|
| Severe Generator Lag | `> 2.0` | Generator loss is much higher | LR × 0.5, batch size −32 |
| Moderate Generator Lag | `> 1.5` | Generator is falling behind | LR × 0.5 |
| Moderate Discriminator Lag | `< 0.67` | Discriminator is falling behind | LR × 1.1 |
| Severe Discriminator Lag | `< 0.50` | Discriminator loss is much higher | LR × 1.1, batch size +32 |
| Balanced | `0.67–1.5` | Training is considered stable | No change |

## Supported Datasets and Models

### MNIST

Uses a fully connected GAN for generating 28×28 grayscale handwritten digits.

### Fashion-MNIST

Uses a DCGAN for generating 28×28 grayscale fashion images.

### CelebA

Uses a DCGAN for generating 64×64 RGB face images. The training pipeline includes flexible dataset-path handling and image preprocessing.

## Repository Structure

```text
gan_agent_project/
│
├── main.py
├── README.md
│
├── agents/
│   ├── hyperparameter_agent.py
│   ├── latent_agent.py
│   └── stop_agent.py
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── dcgan.py
│
├── training/
│   ├── train_mnist_gan.py
│   ├── train_fashion_dcgan.py
│   └── train_celeba_gan.py
│
├── utils/
│   ├── metrics.py
│   └── visualize.py
│
├── data/
│   ├── mnist/
│   ├── fashion-mnist/
│   └── celeba/
│
└── outputs/
    ├── mnist_digits/
    ├── fashion_clothes/
    ├── celeba_faces/
    └── logs/
        └── agent.log
```

## Project Structure

### `agents/`

Contains the components responsible for making training decisions.

**`hyperparameter_agent.py`**

Tracks generator and discriminator losses over a moving window and calculates a smoothed loss ratio. It can update optimizer learning rates and training batch sizes when an imbalance is detected.

The default configuration uses:

- `history_window = 5`
- `cooldown_steps = 3`

The cooldown period gives the previous adjustment time to take effect before another change is made.

**`latent_agent.py`**

Adds controlled Gaussian noise to the latent vector:

```text
noise_scale = 0.05
```

The resulting values are clamped to keep the perturbations within a controlled range.

**`stop_agent.py`**

Evaluates FID after each epoch and generates a stop decision when the target FID is reached or when there has been insufficient improvement for the configured patience period.

### `models/`

Contains the neural network implementations.

The MNIST GAN uses fully connected Generator and Discriminator networks with components such as:

- `LeakyReLU`
- `BatchNorm1d`
- `Dropout`
- `Tanh`

The DCGAN implementation uses convolutional architectures based on:

- `Conv2d`
- `ConvTranspose2d`
- `BatchNorm2d`

The DCGAN weights use the standard initialization with mean `0` and standard deviation `0.02`.

### `training/`

Contains dataset-specific training pipelines.

The MNIST training script:

1. Loads MNIST through TorchVision.
2. Initializes the Generator and Discriminator.
3. Creates the `HyperparameterAgent`.
4. Trains the GAN.
5. Applies agent-based adjustments during training.
6. Saves generated samples.

The Fashion-MNIST and CelebA scripts provide similar training pipelines adapted to their respective image formats and architectures.

### `utils/`

Contains evaluation and visualization utilities.

`metrics.py` provides:

- FID calculation using `pytorch-fid`
- Inception Score using Inception-v3
- Metric history collection

`visualize.py` provides:

- Generated image grids
- Generator/discriminator loss plots

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bhuvantejakomera/Agentic_GAN.git
cd Agentic_GAN
```

### 2. Install Dependencies

Make sure Python 3.9 or newer is installed. Install the required packages:

```bash
pip install torch torchvision numpy pytorch-fid tqdm pillow matplotlib
```


## Running the Project

The easiest way to run the project is through `main.py`.

### MNIST GAN

```bash
python main.py mnist
```

### Fashion-MNIST DCGAN

```bash
python main.py fashion
```

### CelebA DCGAN

```bash
python main.py celeba
```

### Run All Training Pipelines

```bash
python main.py all
```

## Agent Log Demo

If you want to test the agent decision and logging system without running the complete training process:

```bash
python main.py mnist --agent-log-demo
```

The generated log is stored at:

```text
outputs/logs/agent.log
```

## Custom Training

Training scripts can also be executed directly when more control over the configuration is required.

### MNIST

```bash
python training/train_mnist_gan.py \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.0002 \
    --latent-dim 100 \
    --sample-interval 2 \
    --output-dir outputs/mnist_custom
```

### Fashion-MNIST

```bash
python training/train_fashion_dcgan.py \
    --epochs 25 \
    --batch-size 64 \
    --lr 0.0002 \
    --use-cpu
```

### CelebA

First, validate the dataset:

```bash
python training/train_celeba_gan.py \
    --data-dir data/celeba \
    --check-data-only
```

Then start training:

```bash
python training/train_celeba_gan.py \
    --data-dir data/celeba \
    --epochs 40 \
    --image-size 64 \
    --ngf 64 \
    --ndf 64
```

## Outputs

Training results are stored under the `outputs/` directory.

### Generated Samples

Generated images are saved for individual epochs:

```text
outputs/<project>/samples/epoch_XXX.png
```

### Model Checkpoints

Completed models are saved under:

```text
outputs/<project>/checkpoints/
```

Typical checkpoint files include:

```text
generator_<project>.pt
discriminator_<project>.pt
```

### Agent Logs

Agent decisions and parameter adjustments are recorded in:

```text
outputs/logs/agent.log
```

These logs make it possible to inspect why and when the training parameters were changed.

## Evaluation

The framework includes several metrics for monitoring training:

- **FID (Fréchet Inception Distance):** Used to evaluate the similarity between real and generated image distributions.
- **Inception Score:** Used to evaluate generated image quality and diversity.
- **Generator Loss:** Tracks the Generator's training behaviour.
- **Discriminator Loss:** Tracks the Discriminator's training behaviour.

These metrics can be used together with the agent logs to understand how the autonomous adjustments affect training.

