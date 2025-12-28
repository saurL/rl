# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**rl** is a Rust-native reinforcement learning library built on the [burn](https://github.com/tracel-ai/burn) deep learning framework. The goal is to provide a unified RL development experience with production-ready SoTA algorithms that work with arbitrary environments, state spaces, and action spaces. The library is currently in early stages and on hiatus.

**Version**: 0.4.0
**Rust Version**: 1.79+
**License**: MIT

## Build and Development Commands

### Building
```bash
# Build the library
cargo build

# Build with all features
cargo build --all-features

# Build for release
cargo build --release
```

### Testing
```bash
# Run all tests (many modules have inline #[cfg(test)] tests)
cargo test

# Run tests for a specific module
cargo test --lib decay
cargo test --lib ring_buffer

# Run tests with specific features
cargo test --features gym
cargo test --all-features
```

### Running Examples
Examples require specific feature combinations (see Cargo.toml):

```bash
# Examples requiring gym + viz features
cargo run --example q_table_frozen_lake --features "gym,viz"
cargo run --example q_table_snake --features "gym,viz"
cargo run --example dqn_cartpole --features "gym,viz"

# Examples requiring only gym feature
cargo run --example ten_armed_testbed --features gym

# Examples requiring no features
cargo run --example policy_iteration_car_rental
cargo run --example sarsa_windy_gridworld
```

### Documentation
```bash
# Generate and open documentation
cargo doc --open --all-features
```

## Architecture Overview

### Core Trait System

The library is built around extensible trait abstractions:

**Environment Traits** ([src/env.rs](src/env.rs)):
- `Environment` - Core MDP interface with associated `State` and `Action` types, methods: `step()`, `reset()`, `random_action()`, `is_active()`
- `DiscreteActionSpace` - Environments with finite, enumerable actions
- `DiscreteStateSpace` - Environments with finite, enumerable states
- `DeterministicModel` - Environments with known deterministic dynamics
- `KnownDynamics` - Environments with known probability distributions

**Algorithm Architecture**:
- Algorithms are generic over environment traits, not concrete types
- Tabular methods require `DiscreteActionSpace` and `DiscreteStateSpace`
- Deep RL methods (DQN) work with continuous spaces via `ToTensor` trait
- All algorithms are backend-agnostic (work with any burn backend: CPU, GPU, etc.)

### Module Organization

**[src/algo/](src/algo/)** - RL Algorithms:
- `tabular/q_table.rs` - Q-learning with HashMap storage
- `dqn.rs` - Deep Q-Network with experience replay and target networks
- `tabular/ucb.rs` - Upper Confidence Bound for bandits

**[src/exploration/](src/exploration/)** - Exploration strategies:
- All strategies return a `Choice` enum (Explore/Exploit)
- `epsilon_greedy.rs` - ε-greedy with time decay
- `softmax.rs` - Boltzmann exploration
- `ucb.rs` - Upper Confidence Bound exploration

**[src/memory/](src/memory/)** - Experience replay:
- `base.rs` - Standard uniform random sampling (uses RingBuffer)
- `prioritized.rs` - Priority-based sampling with sum tree (for DQN)
- `exp.rs` - Core types: `Exp<E>` (single experience), `ExpBatch<E>` (batched)

**[src/decay.rs](src/decay.rs)** - Hyperparameter decay:
- `Decay` trait with implementations: Constant, Exponential, InverseTime, Linear, Step
- Used for epsilon decay, learning rate schedules, etc.

**[src/ds/](src/ds/)** - Data structures:
- `ring_buffer.rs` - Fixed-size circular buffer (for replay memory)
- `sum_tree.rs` - O(log n) priority sampling tree (for prioritized replay)

**[src/gym/](src/gym/)** - Testing environments (feature-gated: `gym`):
- Each implements `Environment` and relevant sub-traits
- Examples: frozen_lake, cart_pole, k_armed_bandit, windy_gridworld

**[src/viz/](src/viz/)** - Training visualization TUI (feature-gated: `viz`):
- Real-time terminal dashboard with plots, heatmaps, logs
- Runs in separate thread, updates via channels
- Uses ratatui + tui-logger

**[src/traits/](src/traits/)** - Library-specific traits:
- `to_tensor.rs` - `ToTensor` trait for converting states/actions to burn tensors
- `bool_to_tensor.rs` - `BoolToTensor` trait for boolean masks

### Type Safety and Generics

The library heavily leverages Rust's type system:

1. **Backend Abstraction**: Algorithms are generic over burn backends (B: Backend), enabling CPU/GPU flexibility
2. **Dimension Safety**: Tensor operations use const generics for compile-time dimension checking
3. **State/Action Types**: Environments define associated types ensuring compile-time compatibility
4. **Trait Bounds**: Agents require specific environment traits (e.g., DQN needs `DiscreteActionSpace`)

### Key Implementation Patterns

**DQN Agent** ([src/algo/dqn.rs](src/algo/dqn.rs)):
- Uses `DQNModel` trait for neural network definition (implement this for custom networks)
- Supports both standard and prioritized experience replay
- Soft target network updates: target ← τ*online + (1-τ)*target
- AdamW optimizer with configurable gradient clipping

**Q-Table Agent** ([src/algo/tabular/q_table.rs](src/algo/tabular/q_table.rs)):
- HashMap-based Q-value storage for (state, action) pairs
- Epsilon-greedy exploration
- Standard Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Report System**:
- Agents return `Report` (BTreeMap wrapper) from training methods
- Enables consistent metric tracking across algorithms
- Used by viz module for real-time plotting

## Feature Flags

- `gym` - Include testing environments (depends on gym-rs)
- `viz` - Include TUI visualization (depends on ratatui, crossterm, tui-logger)

Most development work should use `--all-features` or `--features "gym,viz"`.

## Important Notes

- **Tests are inline**: Use `#[cfg(test)]` modules within source files, not a separate tests/ directory
- **Examples demonstrate usage**: Check [examples/](examples/) for algorithm + environment combinations
- **Backend choice**: Dev dependencies include both "wgpu" and "ndarray" backends for burn
- **Project status**: Currently on hiatus but accepting contributors
