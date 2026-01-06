//! Deep Q-Network (DQN)
//!
//! DQN is a value-based algorithm that learns to play games and control tasks using
//! deep neural networks to approximate the Q-function for discrete action spaces.
//!
//! # Algorithm Overview
//!
//! DQN learns an action-value function Q(s,a) that estimates the expected return of
//! taking action a in state s. The agent then acts greedily by selecting the action
//! with the highest Q-value.
//!
//! ## Key Features
//! - **Experience replay**: Stores transitions and samples randomly to break correlations
//! - **Target network**: Uses a slowly-updated copy of Q-network for stable learning
//! - **ε-greedy exploration**: Balances exploration and exploitation
//! - **Optional prioritized replay**: Samples important transitions more frequently
//!
//! # Usage Example
//!
//! ## 1. Implementing the Q-Network
//!
//! Create a neural network that implements the `DQNModel` trait:
//!
//! ```rust,ignore
//! use burn::{
//!     config::Config,
//!     module::Module,
//!     nn::{Linear, LinearConfig, Relu},
//!     tensor::backend::AutodiffBackend,
//!     prelude::*,
//! };
//! use rl::algo::dqn::DQNModel;
//!
//! #[derive(Module, Debug)]
//! pub struct QNetwork<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     q_head: Linear<B>,  // Output dimension = number of actions
//! }
//!
//! impl<B: AutodiffBackend> DQNModel<B, 2> for QNetwork<B> {
//!     fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
//!         let x = Relu.forward(self.fc1.forward(state));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         self.q_head.forward(x)  // Q-values for each action
//!     }
//!
//!     fn soft_update(&mut self, other: &Self, tau: f32) {
//!         // Update target network: θ' ← τθ + (1-τ)θ'
//!         soft_update_linear(&mut self.fc1, &other.fc1, tau);
//!         soft_update_linear(&mut self.fc2, &other.fc2, tau);
//!         soft_update_linear(&mut self.q_head, &other.q_head, tau);
//!     }
//! }
//! ```
//!
//! ## 2. Creating an Environment
//!
//! Your environment must implement `Environment` with discrete actions:
//!
//! ```rust,ignore
//! use rl::env::{Environment, DiscreteActionSpace};
//!
//! #[derive(Clone, Copy)]
//! enum Action {
//!     Up = 0,
//!     Down = 1,
//!     Left = 2,
//!     Right = 3,
//! }
//!
//! struct GridWorld {
//!     state: [f32; 2],  // [x, y] position
//!     // ... other fields
//! }
//!
//! impl Environment for GridWorld {
//!     type State = [f32; 2];
//!     type Action = Action;
//!
//!     fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
//!         // Apply action, return (next_state, reward)
//!         // Return None when episode ends
//!     }
//!
//!     fn reset(&mut self) -> Self::State {
//!         // Reset and return initial state
//!     }
//!
//!     fn random_action(&self) -> Self::Action {
//!         // Return random action
//!     }
//! }
//!
//! impl DiscreteActionSpace for GridWorld {
//!     fn actions(&self) -> Vec<Self::Action> {
//!         vec![Action::Up, Action::Down, Action::Left, Action::Right]
//!     }
//! }
//! ```
//!
//! ## 3. Training the Agent
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, Wgpu};
//! use rl::algo::dqn::{DQNAgent, DQNAgentConfig};
//! use rl::decay::ExponentialDecay;
//!
//! type Backend = Autodiff<Wgpu>;
//!
//! fn main() {
//!     let device = Default::default();
//!     let mut env = GridWorld::new();
//!
//!     // Create Q-network
//!     let state_dim = 2;
//!     let num_actions = 4;
//!     let model = QNetworkConfig::new(state_dim, num_actions)
//!         .with_hidden_size(64)
//!         .init::<Backend>(&device);
//!
//!     // Configure agent
//!     let config = DQNAgentConfig {
//!         gamma: 0.99,                    // Discount factor
//!         tau: 0.005,                     // Target network update rate
//!         lr: 1e-3,                       // Learning rate
//!         memory_capacity: 10_000,        // Replay buffer size
//!         memory_batch_size: 32,          // Batch size for learning
//!         use_prioritized_memory: false,  // Use prioritized experience replay
//!         learning_starts: 1000,          // Steps before learning starts
//!         gradient_clip: Some(1.0),       // Gradient clipping
//!         target_update_interval: 100,    // Steps between target updates
//!         ..Default::default()
//!     };
//!
//!     let mut agent = DQNAgent::new(model, config, &device);
//!
//!     // Train
//!     for episode in 0..1000 {
//!         agent.go(&mut env);
//!         println!("Episode {}: reward = {}", episode, env.report.get("reward"));
//!     }
//! }
//! ```
//!
//! # Hyperparameter Tuning
//!
//! ## Common Settings
//! - **Learning rate**: 1e-4 to 1e-3
//! - **Discount (γ)**: 0.95-0.99
//! - **Replay buffer**: 10k-1M transitions
//! - **Batch size**: 32-128
//! - **Target update**: Every 100-1000 steps or soft update with τ=0.001-0.01
//! - **Epsilon decay**: Start at 1.0, decay to 0.01-0.1
//!
//! ## Exploration
//! - Use ε-greedy with exponential or linear decay
//! - Keep minimum epsilon for continued exploration
//! - Consider UCB or Boltzmann exploration for alternatives
//!
//! # Tips
//! - Normalize state inputs to [-1, 1] or [0, 1]
//! - Use reward clipping for stability
//! - Monitor Q-value magnitudes - they shouldn't explode
//! - Prioritized replay can improve sample efficiency
//! - Double DQN (using online net to select, target to evaluate) reduces overestimation
//!
//! Reference: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};

use crate::{
    decay::{self, Decay},
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    memory::{Exp, Memory, PrioritizedReplayMemory, ReplayMemory},
    nn::MLP,
    traits::{BoolToTensor, ToTensor},
};

/// A burn module used with a Deep Q network agent
///
/// ### Generics
/// - `B` - A burn backend
/// - `D` - The dimension of the input tensor
pub trait DQNModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass through the model
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, 2>;

    /// Soft update the parameters of the target network
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    ///
    /// ```ignore
    /// target_net.soft_update(&policy_net, tau);
    /// ```
    fn soft_update(&mut self, other: &Self, tau: f32);
}

/// Configuration for the [`DQNAgent`]
#[derive(Debug, Clone)]
pub struct DQNAgentConfig<D> {
    /// The capacity of the replay memory
    ///
    /// **Default:** `16384`
    pub memory_capacity: usize,
    /// The size of batches to be sampled from the replay memory
    ///
    /// **Default:** `128`
    pub memory_batch_size: usize,
    /// Use [`PrioritizedReplayMemory`] instead of the base [`ReplayMemory`]
    ///
    /// **Default:** `false`
    pub use_prioritized_memory: bool,
    /// The number of episode this agent is going to be trained for
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `500`
    pub num_episodes: usize,
    /// The prioritization exponent, which affects degree of prioritization used in the stochastic sampling of experiences (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `0.7`
    pub prioritized_memory_alpha: f32,
    /// The initial value for beta, the importance sampling exponent, which is annealed from β<sub>0</sub> to 1 to apply IS weights to the temporal difference errors
    /// (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `0.5`
    pub prioritized_memory_beta_0: f32,
    /// The epsilon decay strategy
    ///
    /// **Default:** [`Exponential`](decay::Exponential) decay with decay rate `1e-3`, start value `1.0`, and end value `0.05`
    pub epsilon_decay_strategy: D,
    /// The discount factor
    ///
    /// **Default:** `0.999`
    pub gamma: f32,
    /// The interval at which to perform soft updates on the target network
    ///
    /// **Default:** `1`
    pub target_update_interval: usize,
    /// The rate at which the target network's parameters are soft updated with the policy network's parameters
    ///
    /// **Default:** `5e-3`
    pub tau: f32,
    /// The learning rate for the optimizer
    ///
    /// **Default:** `1e-3`
    pub lr: f32,
}

impl Default for DQNAgentConfig<decay::Exponential> {
    fn default() -> Self {
        Self {
            memory_capacity: 16384,
            memory_batch_size: 128,
            use_prioritized_memory: false,
            num_episodes: 500,
            prioritized_memory_alpha: 0.7,
            prioritized_memory_beta_0: 0.5,
            epsilon_decay_strategy: decay::Exponential::new(1e-3, 1.0, 0.05).unwrap(),
            gamma: 0.999,
            target_update_interval: 1,
            tau: 5e-3,
            lr: 1e-3,
        }
    }
}

/// A Deep Q Network agent
///
/// ### Generics
/// - `B` - A burn backend
/// - `M` - The [`DQNModel`] used for the policy and target networks
/// - `E` - The [`Environment`] in which the agent will learn
///     - The environment's action space must be discrete, since the policy network produces a Q value for each action.
///     - The state and action types' implementations of [`Clone`] should be very lightweight, as they are cloned often.
///       Ideally, both types are [`Copy`].
/// - `DEC` - The decay strategy for epsilon-greedy exploration
/// - `D` - The dimension of the input

pub struct DQNAgent<B, M, E, DEC, const D: usize>
where
    B: AutodiffBackend,
    E: Environment,
    M: AutodiffModule<B>,
    DEC: Decay,
{
    policy_net: M,
    target_net: M,
    device: &'static B::Device,
    memory: Memory<E>,
    optimizer: Option<OptimizerAdaptor<AdamW, M, B>>,
    exploration: EpsilonGreedy<DEC>,
    gamma: f32,
    target_update_interval: usize,
    tau: f32,
    lr: f32,
    total_steps: u32,
    episodes_elapsed: usize,
}

// Manual Clone implementation because OptimizerAdaptor doesn't implement Clone
impl<B, M, E, DEC, const D: usize> Clone for DQNAgent<B, M, E, DEC, D>
where
    B: AutodiffBackend,
    M: DQNModel<B, D> + Clone,
    E: Environment + Clone,
    E::State: Clone,
    E::Action: Clone,
    DEC: Decay + Clone,
{
    fn clone(&self) -> Self {
        // Recreate optimizer instead of cloning it (OptimizerAdaptor doesn't implement Clone)
        let optimizer = Some(
            AdamWConfig::new()
                .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
                .init(),
        );

        Self {
            policy_net: self.policy_net.clone(),
            target_net: self.target_net.clone(),
            device: self.device,
            memory: self.memory.clone(),
            optimizer,
            exploration: self.exploration.clone(),
            gamma: self.gamma,
            target_update_interval: self.target_update_interval,
            tau: self.tau,
            lr: self.lr,
            total_steps: self.total_steps,
            episodes_elapsed: self.episodes_elapsed,
        }
    }
}

impl<B, M, E, DEC, const D: usize> DQNAgent<B, M, E, DEC, D>
where
    B: AutodiffBackend<FloatElem = f32>,
    M: DQNModel<B, D>,
    E: Environment,
    DEC: Decay,
    Vec<E::State>: ToTensor<B, D, Float>,
    Vec<E::Action>: ToTensor<B, 2, Int>,
    E::Action: From<usize>,
    usize: TryFrom<B::IntElem>,
    <usize as TryFrom<B::IntElem>>::Error: std::fmt::Debug,
{
    /// Initialize a new `DQNAgent`
    ///
    /// ### Arguments
    /// - `model` A [`DQNModel`] to be used as the policy and target networks
    /// - `config` A [`DQNAgentConfig`] containing components and hyperparameters for the agent
    /// - `device` A static reference to the device used for the `model`
    pub fn new(model: M, config: DQNAgentConfig<DEC>, device: &'static B::Device) -> Self {
        let model_clone = model.clone();
        let memory = if config.use_prioritized_memory {
            Memory::Prioritized(PrioritizedReplayMemory::new(
                config.memory_capacity,
                config.memory_batch_size,
                config.prioritized_memory_alpha,
                config.prioritized_memory_beta_0,
                config.num_episodes,
            ))
        } else {
            Memory::Base(ReplayMemory::new(
                config.memory_capacity,
                config.memory_batch_size,
            ))
        };

        // ✅ OPTIMISATION : Créer l'optimizer UNE SEULE FOIS et le rendre persistant
        let optimizer = Some(
            AdamWConfig::new()
                .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
                .init(),
        );

        Self {
            policy_net: model,
            target_net: model_clone,
            device,
            memory,
            optimizer,
            exploration: EpsilonGreedy::new(config.epsilon_decay_strategy),
            gamma: config.gamma,
            target_update_interval: config.target_update_interval,
            tau: config.tau,
            lr: config.lr,
            total_steps: 0,
            episodes_elapsed: 0,
        }
    }

    /// Invoke the agent's policy along with the exploration strategy to choose an action from the given state
    fn act(&self, env: &E, state: E::State) -> E::Action {
        match self.exploration.choose(self.total_steps) {
            Choice::Explore => env.random_action(),
            Choice::Exploit => {
                let input = vec![state].to_tensor(self.device);
                let output = self
                    .policy_net
                    
                    .forward(input)
                    .argmax(1)
                    .into_scalar();
                E::Action::from(output.try_into().unwrap())
            }
        }
    }

    /// Perform one DQN learning step
    ///

    fn learn(&mut self) {
        // Sample a batch of memories to train on
        let Memory::Base(memory) = &mut self.memory else {
            return;
        };
        let Some(batch) = memory.sample_zipped() else {
            return;
        };
        let batch_size = memory.batch_size;

        // Create a boolean mask for non-terminal next states so tensor shapes can match in the Bellman Equation
        let non_terminal_mask = batch
            .next_states
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .to_bool_tensor(self.device)
            .unsqueeze_dim::<2>(1);

        // Tensor conversions
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let next_states = batch
            .next_states
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_tensor(self.device);
        let rewards = batch.rewards.to_tensor(self.device).unsqueeze_dim::<2>(1);


        // Compute the Q values of the chosen actions in each state
        let q_values = self.policy_net.forward(states).gather(1, actions);

        // Compute the maximum Q values obtainable from each next state
        let expected_q_values = Tensor::zeros([batch_size, 1], self.device).mask_where(
            non_terminal_mask,
            self.target_net.forward(next_states).max_dim(1).detach(),
        );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);


        let grads = {
            let loss = MseLoss::new().forward(q_values, discounted_expected_return, Reduction::Mean);
            let grads = loss.backward();
            grads
        };

        // Extraire l'optimizer temporairement pour l'update
        let mut optimizer = self.optimizer.take().unwrap();

        // Perform backpropagation on policy net
        let grads_params = GradientsParams::from_grads(grads, &self.policy_net);
        self.policy_net = optimizer.step(self.lr.into(), self.policy_net.clone(), grads_params);

        // Remettre l'optimizer
        self.optimizer = Some(optimizer);

        // Perform a periodic soft update on the parameters of the target network for stable convergence
        if self.episodes_elapsed % self.target_update_interval == 0 {
            self.target_net.soft_update(&self.policy_net, self.tau);
        }
    }

    /// Perform one DQN learning step with prioritized experience replay
    ///

    fn learn_prioritized(&mut self) {
        // Sample a batch of memories to train on
        let Memory::Prioritized(memory) = &mut self.memory else {
            return;
        };
        let Some((batch, weights, indices)) = memory.sample_zipped(self.episodes_elapsed) else {
            return;
        };
        let batch_size = memory.batch_size;

        // Create a boolean mask for non-terminal next states so tensor shapes can match in the Bellman Equation
        let non_terminal_mask = batch
            .next_states
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .to_bool_tensor(self.device)
            .unsqueeze_dim::<2>(1);

        // Tensor conversions
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let next_states = batch
            .next_states
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_tensor(self.device);
        let rewards = batch.rewards.to_tensor(self.device).unsqueeze_dim::<2>(1);

    

        // Compute the Q values of the chosen actions in each state
        let q_values = self.policy_net.forward(states).gather(1, actions);

        // Compute the maximum Q values obtainable from each next state
        let expected_q_values = Tensor::zeros([batch_size, 1], self.device).mask_where(
            non_terminal_mask,
            self.target_net.forward(next_states).max_dim(1).detach(),
        );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        
        let (grads, td_errors) = {
            let tde: Tensor<B, 1> = (discounted_expected_return - q_values).squeeze();

            // Extraire les TD errors AVANT de calculer loss
            let data = tde.to_data();
            let td_errors: Vec<f32> = data.iter::<f32>().collect();

            let weights_tensor = weights.to_tensor(self.device);
            let loss = (weights_tensor * tde.powf_scalar(2.0)).mean();

            let grads = loss.backward();
            


            (grads, td_errors)
        };

        // Update priorities of sampled experiences
        memory.update_priorities(&indices, &td_errors);

        // Extraire l'optimizer temporairement
        let mut optimizer = self.optimizer.take().unwrap();

        // Perform backpropagation on policy net
        let grads_params = GradientsParams::from_grads(grads, &self.policy_net);
        self.policy_net = optimizer.step(self.lr.into(), self.policy_net.clone(), grads_params);

        // Remettre l'optimizer
        self.optimizer = Some(optimizer);

        // Perform a periodic soft update on the parameters of the target network for stable convergence
        if self.episodes_elapsed % self.target_update_interval == 0 {
            self.target_net.soft_update(&self.policy_net, self.tau);
        }
    }

    /// Deploy the `DQNAgent` into the environment for one episode
    ///
    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.act(env, state.clone());
            let (next, reward) = env.step(action.clone());
            next_state = next;

            let exp = Exp {
                state,
                action,
                reward,
                next_state: next_state.clone(),
            };

            match &mut self.memory {
                Memory::Base(memory) => {
                    memory.push(exp);
                    self.learn();
                }
                Memory::Prioritized(memory) => {
                    memory.push(exp);
                    self.learn_prioritized();
                }
            }

            self.total_steps += 1;
        }

        self.episodes_elapsed += 1;
    }
}

// ============================================================================
// MLP Implementations for DQN
// ============================================================================

/// Implementation for 1D tensors (single state)
/// `[features]` → `[1, actions]`
impl<B: AutodiffBackend> DQNModel<B, 1> for MLP<B> {
    fn forward(&self, state: Tensor<B, 1>) -> Tensor<B, 2> {
        let batched = state.unsqueeze_dim(0); // [features] → [1, features]
        MLP::forward(self, batched) // [1, features] → [1, actions]
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}

/// Implementation for 2D tensors (batch of states)
/// `[batch, features]` → `[batch, actions]`
impl<B: AutodiffBackend> DQNModel<B, 2> for MLP<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward(self, state)
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}

/// Implementation for 3D tensors (sequences or multi-entity)
/// `[batch, seq/entities, features]` → `[batch*seq, actions]` then reshape to `[batch, seq*actions]`
impl<B: AutodiffBackend> DQNModel<B, 3> for MLP<B> {
    fn forward(&self, state: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq, features] = state.dims();

        // Flatten: [batch, seq, features] → [batch*seq, features]
        let reshaped = state.reshape([batch * seq, features]);

        // Forward: [batch*seq, features] → [batch*seq, actions]
        let output = MLP::forward::<2>(self, reshaped);

        // Reshape: [batch*seq, actions] → [batch, seq*actions]
        let [_, actions] = output.dims();
        output.reshape([batch, seq * actions])
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}