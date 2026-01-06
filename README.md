# rl - A rust reinforcement learning library

[![Current Crates.io Version](https://img.shields.io/crates/v/rl.svg)](https://crates.io/crates/rl)
[![Documentation](https://img.shields.io/badge/Docs-latest-blue)](https://docs.rs/rl/0.4.0/rl/)
[![Rust Version](https://img.shields.io/badge/Rust-v1.79.0+-tan)](https://releases.rs/docs/1.79.0)

*NOTE: I am currently busy with other tasks, so this project is on hiatus. Development will resume soon.*

## About
**rl** is a fully Rust-native reinforcement learning library with the goal of providing a unified RL development experience, aiming to do for RL what libraries like PyTorch did for deep learning. By leveraging Rust's powerful type system and the [**burn**](https://github.com/tracel-ai/burn) library, **rl** enables users to reuse production-ready SoTA algorithms with arbitrary environments, state spaces, and action spaces. 

This project also aims to provide a clean platform for experimentation with new RL algorithms. By combining **burn**'s powerful deep learning features with **rl**'s provided RL sub-algorithms and components, users can create, test, and benchmark their own new experimental agents without having to start from scratch.

Currently, **rl** is in its early stages. Contributors are more than welcome!

## Features
 - **Simple API**: No need to implement custom model classes - just configure network architecture
 - **High-performance**: Production-ready implementations of SoTA RL algorithms (A2C, DQN, PPO, SAC, TD3)
 - **Visualization**: Real-time training visualization TUI (see image below)
 - **Extensible**: Maximum flexibility for creating and testing new experimental algorithms
 - **Gym environments**: Built-in testing environments
 - **Beginner-friendly**: Comfortable learning experience for those new to RL
 - **Utilities**: General RL peripherals and utility functions

![TUI example](https://github.com/benbaarber/rl/assets/6320364/d0c545bb-a5f4-4487-8e33-1a02a3fb4577)

## Quick Start

Training a DQN agent on CartPole is as simple as:

```rust
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use rl::{
    algo::dqn::{DQNAgent, DQNAgentConfig},
    gym::CartPole,
    nn::MLPConfig,
};

type Backend = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();
    let mut env = CartPole::new(gym_rs::utils::renderer::RenderMode::Human);

    // Just configure the network architecture - no custom models needed!
    let model = MLPConfig::new(4, vec![64, 128], 2).init::<Backend>(&device);

    let config = DQNAgentConfig::default();
    let mut agent = DQNAgent::new(model, config, &device);

    // Train for 500 episodes
    for episode in 0..500 {
        agent.go(&mut env);

        if (episode + 1) % 50 == 0 {
            println!("Episode {}/500", episode + 1);
        }
    }

    println!("Training complete!");
}
```

That's it! No need to implement custom model classes or write boilerplate code. The library provides sensible defaults while remaining fully customizable.

### More Examples

- **A2C** (Actor-Critic): See [`examples/a2c_cartpole`](examples/a2c_cartpole/main.rs)
- **PPO** (Proximal Policy Optimization): See [`examples/ppo_cartpole`](examples/ppo_cartpole/main.rs)
- **SAC** (Soft Actor-Critic): See [`examples/sac_pendulum`](examples/sac_pendulum/main.rs)
- **TD3** (Twin Delayed DDPG): See [`examples/td3_pendulum`](examples/td3_pendulum/main.rs)
