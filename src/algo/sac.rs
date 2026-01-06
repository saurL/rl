//! Soft Actor-Critic (SAC)
//!
//! SAC is an off-policy actor-critic algorithm for continuous action spaces that
//! maximizes both expected return and entropy for robust, exploratory policies.
//!
//! # Algorithm Overview
//!
//! SAC learns a stochastic policy that maximizes the sum of expected reward and entropy:
//! J(π) = E[Σ(r_t + α * H(π(·|s_t)))]
//!
//! where H is the entropy of the policy and α is the temperature parameter.
//!
//! ## Components
//! - **Actor network**: Outputs Gaussian policy parameters (mean, log_std)
//! - **Two critic networks**: Estimate Q-values Q₁(s,a) and Q₂(s,a)
//! - **Target critics**: Slowly-updated copies for stable learning
//! - **Automatic temperature tuning**: Adjusts α to match target entropy
//!
//! ## Key Features
//! - **Maximum entropy RL**: Encourages exploration through entropy regularization
//! - **Clipped double Q-learning**: Uses min(Q₁, Q₂) to reduce overestimation
//! - **Stochastic policy**: Naturally explores without added noise
//! - **Automatic tuning**: Can automatically adjust exploration-exploitation tradeoff
//!
//! # Usage Example
//!
//! ## 1. Implementing Networks
//!
//! Create actor and critic networks that implement the required traits:
//!
//! ```rust,ignore
//! use burn::{
//!     config::Config,
//!     module::Module,
//!     nn::{Linear, LinearConfig, Relu},
//!     tensor::backend::AutodiffBackend,
//!     prelude::*,
//! };
//! use rl::algo::sac::{SACActorModel, SACCriticModel};
//!
//! // Actor: outputs mean and log_std for Gaussian policy
//! #[derive(Module, Debug)]
//! pub struct ActorModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     mean_head: Linear<B>,
//!     log_std_head: Linear<B>,
//! }
//!
//! impl<B: AutodiffBackend> SACActorModel<B, 2> for ActorModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
//!         let x = Relu.forward(self.fc1.forward(state));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         let mean = self.mean_head.forward(x.clone());
//!         let log_std = self.log_std_head.forward(x).clamp(-20.0, 2.0);
//!         (mean, log_std)
//!     }
//!
//!     fn sample_action(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
//!         let (mean, log_std) = self.forward(state);
//!         let std = log_std.clone().exp();
//!
//!         // Sample: action = mean + std * ε
//!         let epsilon = Tensor::random_like(&mean, Distribution::Normal(0.0, 1.0));
//!         let action_unbounded = mean.clone() + std.clone() * epsilon;
//!
//!         // Apply tanh squashing
//!         let action = action_unbounded.clone().tanh();
//!
//!         // Compute log probability with tanh correction
//!         // (see implementation for full formula)
//!
//!         (action, log_prob, mean.tanh())
//!     }
//! }
//!
//! // Critic: outputs Q-value for state-action pair
//! #[derive(Module, Debug)]
//! pub struct CriticModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     q_head: Linear<B>,
//! }
//!
//! impl<B: AutodiffBackend> SACCriticModel<B, 2> for CriticModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
//!         let x = Tensor::cat(vec![state, action], 1);
//!         let x = Relu.forward(self.fc1.forward(x));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         self.q_head.forward(x)
//!     }
//!
//!     fn soft_update(&mut self, other: &Self, tau: f32) {
//!         soft_update_linear(&mut self.fc1, &other.fc1, tau);
//!         soft_update_linear(&mut self.fc2, &other.fc2, tau);
//!         soft_update_linear(&mut self.q_head, &other.q_head, tau);
//!     }
//! }
//! ```
//!
//! ## 2. Creating an Environment
//!
//! Your environment must implement `Environment` and `ContinuousActionSpace`:
//!
//! ```rust,ignore
//! use rl::env::{Environment, ContinuousActionSpace};
//!
//! struct PendulumEnv {
//!     state: [f32; 3],  // [cos(θ), sin(θ), θ_dot]
//!     // ... other fields
//! }
//!
//! impl Environment for PendulumEnv {
//!     type State = [f32; 3];
//!     type Action = [f32; 1];  // torque
//!
//!     fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
//!         // Apply action, return (next_state, reward)
//!     }
//!
//!     fn reset(&mut self) -> Self::State {
//!         // Reset environment
//!     }
//!
//!     fn random_action(&self) -> Self::Action {
//!         // Return random action
//!     }
//! }
//!
//! impl ContinuousActionSpace for PendulumEnv {
//!     fn action_dim(&self) -> usize { 1 }
//!     fn action_bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
//!         Some((vec![-2.0], vec![2.0]))
//!     }
//! }
//! ```
//!
//! ## 3. Training the Agent
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, Wgpu};
//! use rl::algo::sac::{SACAgent, SACAgentConfig};
//!
//! type Backend = Autodiff<Wgpu>;
//!
//! fn main() {
//!     let device = Default::default();
//!     let mut env = PendulumEnv::new();
//!
//!     // Create networks
//!     let actor = ActorConfig::new(3, 1).init::<Backend>(&device);
//!     let critic1 = CriticConfig::new(3, 1).init::<Backend>(&device);
//!     let critic2 = CriticConfig::new(3, 1).init::<Backend>(&device);
//!
//!     // Configure agent
//!     let config = SACAgentConfig {
//!         memory_capacity: 100_000,
//!         memory_batch_size: 256,
//!         gamma: 0.99,                    // Discount factor
//!         tau: 0.005,                     // Soft update rate
//!         lr_actor: 3e-4,                 // Actor learning rate
//!         lr_critic: 3e-4,                // Critic learning rate
//!         lr_alpha: 3e-4,                 // Temperature learning rate
//!         auto_alpha: true,               // Automatic temperature tuning
//!         initial_alpha: 0.2,             // Initial temperature
//!         target_entropy: None,           // Auto-set to -action_dim
//!         learning_starts: 1000,          // Steps before learning
//!         gradient_clip: Some(1.0),       // Gradient clipping
//!         ..Default::default()
//!     };
//!
//!     let mut agent = SACAgent::new(actor, critic1, critic2, &env, config, &device);
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
//! - **Replay buffer**: 100k-1M transitions
//! - **Batch size**: 256 (larger than DQN/TD3)
//! - **Learning rates**: 3e-4 for all networks
//! - **Discount (γ)**: 0.99
//! - **Soft update (τ)**: 0.005
//! - **Target entropy**: -action_dim (automatic tuning default)
//!
//! ## Temperature Parameter (α)
//! - **Auto-tuning (recommended)**: Set `auto_alpha: true`, agent learns α
//! - **Fixed**: Set `auto_alpha: false`, manually tune `initial_alpha`
//! - Higher α → more exploration, lower α → more exploitation
//!
//! # Tips
//! - SAC is sample-efficient and stable - good default for continuous control
//! - Automatic temperature tuning usually works well
//! - Use larger batch sizes than TD3/DDPG (256 vs 100)
//! - Monitor alpha value - it should stabilize during training
//! - Tanh squashing in actor limits actions to [-1, 1], scale to environment bounds
//! - SAC naturally explores, no need for action noise
//!
//! Reference: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (Haarnoja et al., 2018)

use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};
use std::collections::HashMap;

use crate::{
    env::{ContinuousActionSpace, Environment},
    memory::{Exp, ExpBatch, Memory, PrioritizedReplayMemory, ReplayMemory},
    nn::MLP,
    traits::{BoolToTensor, ToTensor, TrainableAgent, TrainingMetrics},
};


/// A burn module used with a SAC agent's actor (policy) network
///
/// The actor outputs mean and log_std for a Gaussian policy over continuous actions.
///
/// ### Generics
/// - `B` - A burn autodiff backend
/// - `D` - The dimension of the input state tensor
pub trait SACActorModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass through the model
    ///
    /// Returns: (mean, log_std) tensors for the action distribution
    /// - mean: [batch_size, action_dim]
    /// - log_std: [batch_size, action_dim]
    fn forward(&self, state: &Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>);
}

/// A burn module used with a SAC agent's critic (Q-value) network
///
/// The critic takes state and action as input and outputs a Q-value.
///
/// ### Generics
/// - `B` - A burn autodiff backend
/// - `STATE_DIM` - The dimension of the state input tensor
pub trait SACCriticModel<B: AutodiffBackend, const STATE_DIM: usize>: AutodiffModule<B> {
    /// Forward pass through the model
    ///
    /// Args:
    ///   - state: [batch_size, state_dim]
    ///   - action: [batch_size, action_dim]
    /// Returns: Q-value [batch_size, 1]
    fn forward(&self, state: &Tensor<B, STATE_DIM>, action: &Tensor<B, 2>) -> Tensor<B, 2>;

    /// Soft update the parameters of the target network
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    ///
    /// ```ignore
    /// target_critic.soft_update(&critic, tau);
    /// ```
    fn soft_update(&mut self, other: &Self, tau: f32);
}

/// Configuration for the [`SACAgent`]
#[derive(Debug, Clone)]
pub struct SACAgentConfig {
    // Memory configuration
    /// The capacity of the replay memory
    ///
    /// **Default:** `1_000_000`
    pub memory_capacity: usize,
    /// The size of batches to be sampled from the replay memory
    ///
    /// **Default:** `256`
    pub memory_batch_size: usize,
    /// Use [`PrioritizedReplayMemory`] instead of the base [`ReplayMemory`]
    ///
    /// **Default:** `false`
    pub use_prioritized_memory: bool,
    /// The number of episodes this agent is going to be trained for
    ///
    /// This value is only used if `use_prioritized_memory` is set to true
    ///
    /// **Default:** `1000`
    pub num_episodes: usize,
    /// The prioritization exponent α (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_memory` is set to true
    ///
    /// **Default:** `0.6`
    pub prioritized_memory_alpha: f32,
    /// The initial value for β (importance sampling exponent), annealed from β₀ to 1
    /// (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_memory` is set to true
    ///
    /// **Default:** `0.4`
    pub prioritized_memory_beta_0: f32,

    // SAC hyperparameters
    /// The discount factor γ
    ///
    /// **Default:** `0.99`
    pub gamma: f32,
    /// The soft update rate τ for target networks
    ///
    /// **Default:** `0.005`
    pub tau: f32,
    /// The learning rate for the actor network
    ///
    /// **Default:** `3e-4`
    pub lr_actor: f64,
    /// The learning rate for the critic networks
    ///
    /// **Default:** `3e-4`
    pub lr_critic: f64,
    /// The learning rate for alpha (temperature) when using automatic tuning
    ///
    /// **Default:** `3e-4`
    pub lr_alpha: f64,

    // Alpha (temperature) configuration
    /// Use automatic alpha (temperature) tuning
    ///
    /// If true, alpha is learned to match target entropy. If false, alpha is fixed.
    ///
    /// **Default:** `true`
    pub auto_alpha: bool,
    /// The initial/fixed value for alpha (temperature coefficient)
    ///
    /// Used as the starting value when auto_alpha is true, or as the constant value when false.
    ///
    /// **Default:** `0.2`
    pub initial_alpha: f32,
    /// Target entropy for automatic alpha tuning
    ///
    /// If None, defaults to -action_dim (heuristic from SAC paper)
    ///
    /// **Default:** `None`
    pub target_entropy: Option<f32>,

    // Training configuration
    /// Update target networks every N steps
    ///
    /// **Default:** `1`
    pub target_update_interval: usize,
    /// Update actor every N critic updates (delayed policy updates)
    ///
    /// **Default:** `2`
    pub actor_update_interval: usize,
    /// Number of gradient steps per environment step
    ///
    /// **Default:** `1`
    pub gradient_steps: usize,
    /// Number of steps to collect before learning starts
    ///
    /// **Default:** `10000`
    pub learning_starts: usize,
    /// Perform updates every N environment steps (instead of every step)
    ///
    /// Setting this to N > 1 amortizes update costs by batching them.
    /// When combined with gradient_steps > 1, provides better throughput.
    ///
    /// **Default:** `1` (update every step)
    pub update_frequency: usize,
}

impl Default for SACAgentConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 1_000_000,
            memory_batch_size: 256,
            use_prioritized_memory: false,
            num_episodes: 1000,
            prioritized_memory_alpha: 0.6,
            prioritized_memory_beta_0: 0.4,
            gamma: 0.99,
            tau: 0.005,
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            lr_alpha: 3e-4,
            auto_alpha: true,
            initial_alpha: 0.2,
            target_entropy: None,
            target_update_interval: 1,
            actor_update_interval: 2,
            gradient_steps: 1,
            learning_starts: 10000,
            update_frequency: 1,
        }
    }
}

/// A Soft Actor-Critic (SAC) agent for continuous control
///
/// SAC is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework.
/// It learns a stochastic policy that maximizes both expected return and entropy.
///
/// ### Features
/// - Clipped Double-Q learning to reduce overestimation bias
/// - Tanh squashing for bounded continuous actions
/// - Automatic temperature (α) tuning to match target entropy
/// - Support for Prioritized Experience Replay
///
/// ### Generics
/// - `B` - A burn autodiff backend
/// - `Actor` - Actor network implementing [`SACActorModel`]
/// - `Critic` - Critic network implementing [`SACCriticModel`]
/// - `E` - Environment implementing [`Environment`] and [`ContinuousActionSpace`]
/// - `STATE_DIM` - Dimension of the state input tensor
pub struct SACAgent<B, Actor, Critic, E, const STATE_DIM: usize>
where
    B: AutodiffBackend,
    E: Environment + ContinuousActionSpace,
    Critic: AutodiffModule<B>,
    Actor: AutodiffModule<B>,
{
    // Networks (Option for ownership during optimization)
    actor: Option<Actor>,
    critic1: Option<Critic>,
    critic2: Option<Critic>,
    target_critic1: Option<Critic>,
    target_critic2: Option<Critic>,

    // Alpha (temperature) handling
    log_alpha: Option<Tensor<B, 1>>,
    alpha: f32,
    target_entropy: f32,
    auto_alpha: bool,

    // Memory and device
    memory: Memory<E>,
    device: &'static B::Device,

    // Hyperparameters
    gamma: f32,
    tau: f32,
    lr_actor: f64,
    lr_critic: f64,
    lr_alpha: f64,

    // Training state
    total_steps: usize,
    episodes_elapsed: usize,
    target_update_interval: usize,
    actor_update_interval: usize,
    gradient_steps: usize,
    learning_starts: usize,
    update_count: usize,
    is_learning: bool,
    eval_mode: bool,
    update_frequency: usize,

    // Action space info
    action_dim: usize,
    action_bounds: Option<(Vec<f32>, Vec<f32>)>,

    optimizer_critic1: OptimizerAdaptor<AdamW, Critic, B>,
    optimizer_critic2: OptimizerAdaptor<AdamW, Critic, B>,
    optimizer_actor: OptimizerAdaptor<AdamW, Actor, B>,
    optimizer_alpha: OptimizerAdaptor<AdamW, Tensor<B, 1>, B>,
}

impl<B, Actor, Critic, E, const STATE_DIM: usize> SACAgent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: SACActorModel<B, STATE_DIM>,
    Critic: SACCriticModel<B, STATE_DIM>,
    E: Environment + ContinuousActionSpace,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
    Vec<E::Action>: ToTensor<B, 2, Float>,
{
    const LOG_STD_MIN: f32 = -20.0;  // Quasi-déterministe
    const LOG_STD_MAX: f32 = 2.0;     // Exploration max

    /// Get the current size of the replay buffer
    pub fn buffer_size(&self) -> usize {
        match &self.memory {
            Memory::Base(mem) => mem.len(),
            Memory::Prioritized(mem) => mem.len(),
        }
    }

    /// Create a new SAC agent
    ///
    /// # Arguments
    /// - `actor` - The actor (policy) network
    /// - `critic1` - The first critic network
    /// - `critic2` - The second critic network
    /// - `env` - A reference to the environment (to get action space info)
    /// - `config` - Configuration for the agent
    /// - `device` - The device to run computations on
    pub fn new(
        actor: Actor,
        critic1: Critic,
        critic2: Critic,
        env: &E,
        config: SACAgentConfig,
        device: &'static B::Device,
    ) -> Self {
        
        let action_dim = env.action_dim();
        let action_bounds = env.action_bounds();

        // Initialize target critics as copies of online critics
        let target_critic1 = critic1.clone();
        let target_critic2 = critic2.clone();

        // Calculate target entropy if not provided
        let target_entropy = config
            .target_entropy
            .unwrap_or_else(|| -(action_dim as f32));

        // Initialize log_alpha as a learnable parameter if auto-tuning
        let log_alpha = if config.auto_alpha {
            let alpha_value = config.initial_alpha.ln();
            Some(
                Tensor::<B, 1>::from_data(
                    TensorData::from([alpha_value]).convert::<B::FloatElem>(),
                    device,
                )
                .require_grad(),
            )
        } else {
            None
        };

        // Create memory
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
        let optimizer_critic1 = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let optimizer_critic2 = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let optimizer_actor = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let optimizer_alpha = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        Self {
            actor: Some(actor),
            critic1: Some(critic1),
            critic2: Some(critic2),
            target_critic1: Some(target_critic1),
            target_critic2: Some(target_critic2),
            log_alpha,
            alpha: config.initial_alpha,
            target_entropy,
            auto_alpha: config.auto_alpha,
            memory,
            device,
            gamma: config.gamma,
            tau: config.tau,
            lr_actor: config.lr_actor,
            lr_critic: config.lr_critic,
            lr_alpha: config.lr_alpha,
            total_steps: 0,
            episodes_elapsed: 0,
            target_update_interval: config.target_update_interval,
            actor_update_interval: config.actor_update_interval,
            gradient_steps: config.gradient_steps,
            learning_starts: config.learning_starts,
            update_count: 0,
            is_learning: false,
            eval_mode: false,
            update_frequency: config.update_frequency,
            action_dim,
            action_bounds,
            optimizer_critic1,
            optimizer_critic2,
            optimizer_actor,
            optimizer_alpha,
        }
    }

    /// Compute log probability of actions under the policy
    ///
    /// Implements the full SAC log-probability computation with tanh squashing correction:
    /// log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
    ///
    /// where u is the unbounded action sampled from Gaussian N(mean, std²)
    /// and a = tanh(u) is the bounded action.
    ///
    /// # Arguments
    /// - `mean` - Mean of the Gaussian distribution [batch_size, action_dim]
    /// - `log_std` - Log standard deviation [batch_size, action_dim]
    /// - `unbounded_action` - Sampled unbounded action u [batch_size, action_dim]
    ///
    /// # Returns
    /// Log probability [batch_size]
    fn compute_log_prob(
        &self,
        mean: &Tensor<B, 2>,
        log_std: &Tensor<B, 2>,
        unbounded_action: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Compute Gaussian log probability: log N(u | mean, std²)
        // log p(u) = -0.5 * [(u - mean)² / std² + log(2π) + 2*log(std)]

        let std = log_std.clone().exp();
        let var = std.clone().powf_scalar(2.0);

        // (u - mean)² / var
        let normalized_diff = (unbounded_action.clone() - mean.clone()).powf_scalar(2.0) / var;

        // Full Gaussian log prob per dimension
        let log_2pi_scalar = (2.0 * std::f32::consts::PI).ln();
        let log_std_term = log_std.clone().mul_scalar(2.0);
        let gaussian_log_prob: Tensor<B, 2> = (normalized_diff.add_scalar(log_2pi_scalar) + log_std_term).mul_scalar(-0.5);

        // Sum across action dimensions to get total log prob
        let total_gaussian_log_prob = gaussian_log_prob.sum_dim(1);

        // Tanh squashing correction: -Σ log(1 - tanh²(u))
        // Using identity: log(1 - tanh²(u)) = log(sech²(u)) = -2*log(cosh(u))
        // And stable computation: log(cosh(u)) = |u| + log((1 + exp(-2|u|)) / 2)

        let abs_u = unbounded_action.clone().abs();
        let neg_2abs_u = abs_u.clone().mul_scalar(-2.0);
        let exp_term = neg_2abs_u.exp();
        let log_cosh: Tensor<B, 2> = abs_u + (exp_term.add_scalar(1.0)).div_scalar(2.0).log();
        let tanh_correction = log_cosh.sum_dim(1).mul_scalar(-2.0);

        // Final log probability with correction
        let log_prob = total_gaussian_log_prob + tanh_correction;

        // Return as [batch_size, 1] for compatibility with Q-values
        log_prob.unsqueeze()

   
    }

    /// Sample an action from the policy
    ///
    /// Returns: (mean, log_std, unbounded_action, bounded_action)
    /// - mean: [batch_size, action_dim] - mean of the Gaussian
    /// - log_std: [batch_size, action_dim] - log std of the Gaussian
    /// - unbounded_action: [batch_size, action_dim] - sampled action before tanh (u)
    /// - bounded_action: [batch_size, action_dim] - action after tanh squashing (tanh(u))
    fn sample_action(
        &self,
        actor: &Actor,
        state: &Tensor<B, STATE_DIM>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Get mean and log_std from actor
        let (mean, log_std) = actor.forward(state);

        // Sample from Gaussian: u = mean + std * ε where ε ~ N(0,1)
        let std = log_std.clone().exp();
        let epsilon = Tensor::random_like(&mean, burn::tensor::Distribution::Normal(0.0, 1.0));
        let unbounded_action = mean.clone() + std * epsilon;

        // Apply tanh squashing to get bounded action
        let bounded_action = unbounded_action.clone().tanh();

        (mean, log_std, unbounded_action, bounded_action)
    }

    /// Select an action given a state
    ///
    /// In eval mode: uses mean action (deterministic)
    /// In train mode: samples from policy (stochastic)
    ///
    /// # Arguments
    /// - `state` - The current state
    ///
    /// # Returns
    /// The selected action
    pub fn act(&self, state: &E::State) -> E::Action {
        let deterministic = self.eval_mode;
        self.act_internal(state.clone(), deterministic)
    }

    /// Internal action selection with explicit deterministic flag
    fn act_internal(&self, state: E::State, deterministic: bool) -> E::Action {
        let state_tensor = vec![state].to_tensor(self.device);

        let actor = self.actor.as_ref().unwrap();

        let action_tensor = {

            if deterministic {
                // Use mean action for deterministic evaluation
                let (mean, _) = actor.forward(&state_tensor);

                mean.tanh()
            } else {
                // Sample action for exploration
                let (_mean, _log_std, _unbounded, bounded) = self.sample_action(actor, &state_tensor);
                bounded
            }
        };


        // Extract action and scale to environment bounds
        let action_data = action_tensor.into_data();
        let action_vec: Vec<f32> = action_data.iter().collect();

        // Scale from [-1, 1] to [low, high] if bounds exist
        let scaled_action = if let Some((ref low, ref high)) = self.action_bounds {
            action_vec
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let l = low[i];
                    let h = high[i];
                    (a + 1.0) * 0.5 * (h - l) + l
                })
                .collect::<Vec<_>>()
        } else {
            action_vec
        };

        // Convert Vec<f32> to E::Action
        unsafe {
            let ptr = scaled_action.as_ptr() as *const E::Action;
            ptr.read()
        }
    }

    /// Soft update target critic networks
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    #[inline]
    fn soft_update_targets(&mut self) {
        let critic1 = self.critic1.as_ref().unwrap();
        let critic2 = self.critic2.as_ref().unwrap();

        self.target_critic1.as_mut().unwrap().soft_update(critic1, self.tau);
        self.target_critic2.as_mut().unwrap().soft_update(critic2, self.tau);
    }

    /// Update critic networks with clipped double-Q learning
    ///
    /// # Returns
    /// (td_errors, critic1_loss, critic2_loss)
    /// - td_errors: TD errors for each sample (for prioritized replay)
    /// - critic1_loss: Loss value for critic 1
    /// - critic2_loss: Loss value for critic 2
    fn update_critics(
        &mut self,
        batch: &crate::memory::ExpBatch<E>,
        weights: Option<&Tensor<B, 1>>,
    ) -> (Vec<f32>, f32, f32) {
        let batch_size = batch.states.len();

        // Convert batch to tensors

        let states = batch.states.clone().to_tensor(self.device);
        let actions = batch.actions.clone().to_tensor(self.device);

        // Pre-allocate rewards tensor directly (avoid intermediate allocation)
        let rewards = Tensor::<B, 1>::from_data(
            TensorData::from(batch.rewards.as_slice()).convert::<B::FloatElem>(),
            self.device,
        )
        .reshape([batch_size, 1]);

        // Build mask and next_states in a single pass to avoid double iteration
        let placeholder = &batch.states[0];
        let mut mask_vec = Vec::with_capacity(batch_size);
        let mut next_states_vec = Vec::with_capacity(batch_size);

        for ns in batch.next_states.iter() {
            if let Some(ref state) = ns {
                mask_vec.push(true);
                next_states_vec.push(state.clone());
            } else {
                mask_vec.push(false);
                next_states_vec.push(placeholder.clone());
            }
        }

        let non_terminal_mask = mask_vec
            .to_bool_tensor(self.device)
            .reshape([batch_size, 1]);
        let next_states = next_states_vec.to_tensor(self.device);


        // Compute TD target (everything here should be detached/no_grad)

        let td_target = {
            let target_critic1 = self.target_critic1.as_ref().unwrap();
            let target_critic2 = self.target_critic2.as_ref().unwrap();
            let actor = self.actor.as_ref().unwrap();

            // Sample next actions from current policy
            let (next_mean, next_log_std, next_unbounded, next_bounded) = self.sample_action(actor, &next_states);
            let next_bounded = next_bounded.detach();

            // Compute log probability using SACAgent method
            let next_log_probs = self.compute_log_prob(&next_mean, &next_log_std, &next_unbounded).detach();

            // Compute target Q-values using clipped double-Q
            let target_q1 = target_critic1.forward(&next_states, &next_bounded);
            let target_q2 = target_critic2.forward(&next_states, &next_bounded);
            let target_q = target_q1.min_pair(target_q2);

            // Apply entropy regularization: V(s') = Q(s', a') - α * entropy
            // where entropy = -log π(a'|s'), so V(s') = Q + α * log π
            let entropy_term = next_log_probs * self.alpha;
            let target_value = target_q + entropy_term;
            // Apply non-terminal mask
            let zeros = Tensor::zeros([batch_size, 1], self.device);
            let masked_target = zeros.mask_where(non_terminal_mask, target_value);

            // Compute TD target and detach it (it's a fixed target)
            (rewards + masked_target * self.gamma).detach()
        };


        let loss1_val;
        let loss2_val;

        // Update critic1: forward -> loss -> backward -> optimize
        let (td_error1, critic1_updated) = {

            let critic1 = self.critic1.take().unwrap();
            let q1_pred = critic1.forward(&states, &actions);

            let td_error1 = (q1_pred.clone() - td_target.clone()).detach();

            let loss1 = if let Some(w) = weights {
                let squared_errors: Tensor<B, 1> = (q1_pred - td_target.clone())
                    .powf_scalar(2.0)
                    .squeeze_dims(&[1]);
                (w.clone() * squared_errors).mean()
            } else {
                MseLoss::new().forward(q1_pred, td_target.clone(), Reduction::Mean)
            };
            loss1_val = loss1.clone().into_scalar().elem::<f32>();

            let grads1 = loss1.backward();

            let grads1_params = GradientsParams::from_grads(grads1, &critic1);

            let critic1_updated = self.optimizer_critic1.step(self.lr_critic, critic1, grads1_params);

            (td_error1, critic1_updated)
        };
        self.critic1 = Some(critic1_updated);

        // Update critic2: forward -> loss -> backward -> optimize (completely separate)
        let (td_error2, critic2_updated) = {

            let critic2 = self.critic2.take().unwrap();
            let q2_pred = critic2.forward(&states, &actions);

            let td_error2 = (q2_pred.clone() - td_target.clone()).detach();

            let loss2 = if let Some(w) = weights {
                let squared_errors: Tensor<B, 1> = (q2_pred - td_target.clone())
                    .powf_scalar(2.0)
                    .squeeze_dims(&[1]);
                (w.clone() * squared_errors).mean()
            } else {
                MseLoss::new().forward(q2_pred, td_target, Reduction::Mean)
            };
            loss2_val = loss2.clone().into_scalar().elem::<f32>();

            let grads2 = loss2.backward();

            let grads2_params = GradientsParams::from_grads(grads2, &critic2);

            let critic2_updated = self.optimizer_critic2.step(self.lr_critic, critic2, grads2_params);

            (td_error2, critic2_updated)
        };
        self.critic2 = Some(critic2_updated);

        // Compute average TD errors for PER
        let avg_td_errors: Tensor<B, 1> = ((td_error1.abs() + td_error2.abs()) / 2.0)
            .squeeze_dims(&[1]);

        // Return TD errors and loss values
        let td_errors = avg_td_errors.into_data().iter::<f32>().collect();
        (td_errors, loss1_val, loss2_val)
    }

    /// Update actor network
    ///
    /// Maximize Q - α*log_π and return actor loss and average entropy
    fn update_actor(&mut self, batch: &ExpBatch<E>) -> (f32, f32) {
    let states = batch.states.clone().to_tensor(self.device);
    let actor = self.actor.take().unwrap();
    let critic1 = self.critic1.as_ref().unwrap();
    let critic2 = self.critic2.as_ref().unwrap();

    // Sample actions
    let (mean, log_std, unbounded, bounded) = self.sample_action(&actor, &states);
println!("=== ACTOR DEBUG ===");
    println!("mean: min={:.4}, max={:.4}, mean={:.4}",
             mean.clone().min().into_scalar().elem::<f32>(),
             mean.clone().max().into_scalar().elem::<f32>(),
             mean.clone().mean().into_scalar().elem::<f32>());
    println!("log_std: min={:.4}, max={:.4}, mean={:.4}",
             log_std.clone().min().into_scalar().elem::<f32>(),
             log_std.clone().max().into_scalar().elem::<f32>(),
             log_std.clone().mean().into_scalar().elem::<f32>());

    // Compute log probability
    let log_probs = self.compute_log_prob(&mean, &log_std, &unbounded);
println!("log_probs: min={:.4}, max={:.4}, mean={:.4}",
             log_probs.clone().min().into_scalar().elem::<f32>(),
             log_probs.clone().max().into_scalar().elem::<f32>(),
             log_probs.clone().mean().into_scalar().elem::<f32>());


    // 🔍 DEBUG


    // Compute Q-values
    let q1 = critic1.forward(&states, &bounded);
    let q2 = critic2.forward(&states, &bounded);
    let q_min = q1.min_pair(q2);


println!("q_min: min={:.4}, max={:.4}, mean={:.4}",
             q_min.clone().min().into_scalar().elem::<f32>(),
             q_min.clone().max().into_scalar().elem::<f32>(),
             q_min.clone().mean().into_scalar().elem::<f32>());

    // Actor loss
    let actor_loss = (log_probs.clone() * self.alpha - q_min).mean();
    let actor_loss_val = actor_loss.clone().into_scalar().elem::<f32>();

println!("actor_loss: {:.4}", actor_loss_val);
    println!("==================");

    // Entropy
    let avg_entropy = -log_probs.mean().into_scalar().elem::<f32>();


    // Backprop
    let grads = actor_loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &actor);
    self.actor = Some(self.optimizer_actor.step(self.lr_actor, actor, grads_params));

    (actor_loss_val, avg_entropy)
}

    /// Update alpha (temperature) via automatic tuning, returns alpha loss
    ///
    /// Maximize entropy towards target_entropy
    fn update_alpha(&mut self, batch: &crate::memory::ExpBatch<E>) -> f32 {
        if !self.auto_alpha {
            return 0.0;
        }

        let states = batch.states.clone().to_tensor(self.device);
        let actor = self.actor.as_ref().unwrap();
        let log_alpha = self.log_alpha.take().unwrap();


        // Sample actions to compute entropy (detach to avoid backprop through actor)
        let (mean, log_std, unbounded, _bounded) = self.sample_action(actor, &states);

        // Compute log probability using SACAgent method
        let log_probs = self.compute_log_prob(&mean, &log_std, &unbounded);
        let log_probs = log_probs.detach();

        // Alpha loss: -α * (log_π + target_entropy)
        // This maximizes entropy towards target_entropy
        let alpha_exp = log_alpha.clone().exp();
        let entropy_diff: Tensor<B, 1> = (log_probs + self.target_entropy).squeeze_dims(&[1]);
        let alpha_loss = -(alpha_exp * entropy_diff).mean();
        let alpha_loss_val = alpha_loss.clone().into_scalar().elem::<f32>();


        // Update log_alpha
        let grads = {
            alpha_loss.backward()
        };

        let grads_params = GradientsParams::from_grads(grads, &log_alpha);

        let new_log_alpha = self.optimizer_alpha.step(self.lr_alpha, log_alpha, grads_params);

        // Update alpha value (for use in other loss functions)
        let alpha_data = new_log_alpha.clone().exp().into_data();
        let alpha_scalar: f32 = alpha_data.iter::<f32>().next().unwrap();
        self.alpha = alpha_scalar;

        // Re-register gradients for the next iteration
        self.log_alpha = Some(new_log_alpha.require_grad());

        alpha_loss_val
    }

    /// Main learning loop, returns training metrics if updated
    fn learn_internal(&mut self) -> Option<TrainingMetrics> {
        // Check if we should start learning (only check once)
        if !self.is_learning {
            if self.total_steps >= self.learning_starts {
                self.is_learning = true;
            } else {
                return None;
            }
        }

        let mut total_critic1_loss = 0.0;
        let mut total_critic2_loss = 0.0;
        let mut total_actor_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut total_alpha_loss = 0.0;
        let mut n_updates = 0;

        for _gradient_step in 0..self.gradient_steps {
            // Sample batch (with or without PER)

            let (batch, weights, indices) = match &mut self.memory {
                Memory::Base(memory) => {
                    let batch = memory.sample_zipped();
                    if let Some(b) = batch {
                        (b, None, None)
                    } else {
                        return None;
                    }
                }
                Memory::Prioritized(memory) => {
                    if let Some((batch, weights, indices)) = memory.sample_zipped(self.episodes_elapsed) {
                        let w_tensor = Tensor::<B, 1>::from_data(
                            TensorData::from(weights.as_slice()).convert::<B::FloatElem>(),
                            self.device,
                        );
                        (batch, Some(w_tensor), Some(indices))
                    } else {
                        return None;
                    }
                }
            };


            // 1. Update critics (with clipped double-Q)
            let (td_errors, critic1_loss, critic2_loss) = self.update_critics(&batch, weights.as_ref());

            total_critic1_loss += critic1_loss;
            total_critic2_loss += critic2_loss;
            n_updates += 1;

            // 2. Update actor (delayed policy updates)
            if self.update_count % self.actor_update_interval == 0 {
                let (actor_loss, entropy) = self.update_actor(&batch);
                total_actor_loss += actor_loss;
                total_entropy += entropy;

                // 3. Update alpha (if auto-tuning)
                if self.auto_alpha {
                    let alpha_loss = self.update_alpha(&batch);
                    total_alpha_loss += alpha_loss;
                }
            }

            // 4. Soft update target networks
            if self.update_count % self.target_update_interval == 0 {
                self.soft_update_targets();
            }

            // 5. Update priorities for PER
            if let Some(ref indices_vec) = indices {
                if let Memory::Prioritized(memory) = &mut self.memory {
                    memory.update_priorities(indices_vec, &td_errors);
                }
            }

            self.update_count += 1;
        }

        // Return averaged training metrics
        if n_updates > 0 {
            Some(TrainingMetrics {
                policy_loss: total_actor_loss / n_updates as f32,
                value_loss: (total_critic1_loss + total_critic2_loss) / (2.0 * n_updates as f32),
                entropy: total_entropy / n_updates as f32,
                approx_kl: None,
                clip_fraction: None,
                early_stopped: false,
                n_updates,
                extra: {
                    let mut map = HashMap::new();
                    map.insert("critic1_loss".to_string(), total_critic1_loss / n_updates as f32);
                    map.insert("critic2_loss".to_string(), total_critic2_loss / n_updates as f32);
                    map.insert("alpha".to_string(), self.alpha);
                    map.insert("alpha_loss".to_string(), total_alpha_loss / n_updates as f32);
                    map
                },
            })
        } else {
            None
        }
    }

    /// Train the agent on an environment
    ///
    /// This is the main training loop that collects experiences and learns from them.
    ///
    /// # Arguments
    /// - `env` - The environment to train on
    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());

        // Training loop
        while let Some(state) = next_state {
            // Select action (explore during training)
            let action = self.act_internal(state.clone(), false);

            // Take step in environment
            let (next, reward) = env.step(action.clone());
            
            next_state = next.clone();

            // Create experience and store in memory
            let exp = Exp {
                state,
                action,
                reward,
                next_state: next,
            };

            match &mut self.memory {
                Memory::Base(memory) => memory.push(exp),
                Memory::Prioritized(memory) => memory.push(exp),
            }

            // Learn (only every update_frequency steps for better throughput)
            if self.total_steps % self.update_frequency == 0 {
                self.learn_internal();
            }

            self.total_steps += 1;

            // Check if episode ended
            if next_state.is_none() {
                self.episodes_elapsed += 1;
                break;
            }
        }
    }
}

/// Implementation of TrainableAgent trait for SACAgent
impl<B, Actor, Critic, E, const STATE_DIM: usize> TrainableAgent<E>
    for SACAgent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: SACActorModel<B, STATE_DIM>,
    Critic: SACCriticModel<B, STATE_DIM>,
    E: Environment + ContinuousActionSpace,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
    Vec<E::Action>: ToTensor<B, 2, Float>,
{
    type StepInfo = (f32, bool, Option<TrainingMetrics>);

    fn step(&mut self, env: &mut E) -> Self::StepInfo {
        let state = env.current_state();
        let action = self.act(&state);
        let (next_state_opt, reward) = env.step(action.clone());
        let done = next_state_opt.is_none();

        // Store experience in memory only during training
        if !self.eval_mode {
            let exp = Exp {
                state,
                action,
                reward,
                next_state: next_state_opt.clone(),
            };

            match &mut self.memory {
                Memory::Base(memory) => memory.push(exp),
                Memory::Prioritized(memory) => memory.push(exp),
            }
        }

        self.total_steps += 1;

        // Check if we should start learning
        if !self.is_learning && self.total_steps >= self.learning_starts && !self.eval_mode {
            self.is_learning = true;
        }

        // Learn if conditions are met (only every update_frequency steps)
        let metrics = if self.should_learn() && self.total_steps % self.update_frequency == 0 {
            self.learn_internal()
        } else {
            None
        };

        // Update episodes counter if done
        if done {
            self.episodes_elapsed += 1;
        }

        (reward, done, metrics)
    }

    fn should_learn(&self) -> bool {
        self.is_learning && !self.eval_mode
    }

    fn learn(&mut self, _next_state: &E::State, _done: bool) -> TrainingMetrics {
        // SAC doesn't need next_state or done for learn() since it uses replay memory
        // This method is required by the trait but learn_internal() is called from step()
        self.learn_internal().unwrap_or(TrainingMetrics {
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            approx_kl: None,
            clip_fraction: None,
            early_stopped: false,
            n_updates: 0,
            extra: HashMap::new(),
        })
    }

    fn reset_episode(&mut self) {
        // SAC uses replay memory, so no per-episode reset needed
        // Memory persists across episodes
    }

    fn eval(&mut self) {
        self.eval_mode = true;
    }

    fn train(&mut self) {
        self.eval_mode = false;
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }
}

// ================================================================================================
// MLP Implementations for SAC
// ================================================================================================

// Note: SAC Actor requires outputting both mean and log_std, which is not directly supported
// by a single MLP. Users can either:
// 1. Use a custom actor model that implements SACActorModel
// 2. Create a wrapper struct with two MLPs (one for mean, one for log_std)
//
// For SAC Critic, we can implement the trait for a single MLP that takes concatenated [state, action]

use burn::module::Module;

/// Helper struct to use MLP as SAC Critic
///
/// Concatenates state and action, then passes through a single MLP to get Q-value.
#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    mlp: MLP<B>,
}

impl<B: Backend> MLPCritic<B> {
    /// Create a new MLP-based critic
    ///
    /// # Arguments
    /// - `state_dim` - Dimension of state
    /// - `action_dim` - Dimension of action
    /// - `hidden_layers` - Hidden layer sizes (e.g., [256, 256])
    /// - `device` - Device to create the network on
    pub fn new(state_dim: usize, action_dim: usize, hidden_layers: Vec<usize>, device: &B::Device) -> Self {
        use crate::nn::MLPConfig;
        let mlp = MLPConfig::new(state_dim + action_dim, hidden_layers, 1).init(device);
        Self { mlp }
    }
}

impl<B: AutodiffBackend, const STATE_DIM: usize> SACCriticModel<B, STATE_DIM> for MLPCritic<B> {
    fn forward(&self, state: &Tensor<B, STATE_DIM>, action: &Tensor<B, 2>) -> Tensor<B, 2> {
        // Handle different state dimensions
        let state_2d = match STATE_DIM {
            1 => {
                // Single state [features] -> [1, features]
                let state_ref: &Tensor<B, 1> = unsafe { &*(state as *const _ as *const Tensor<B, 1>) };
                state_ref.clone().unsqueeze_dim(0)
            }
            2 => {
                // Batch of states [batch, features] -> use directly
                let state_ref: &Tensor<B, 2> = unsafe { &*(state as *const _ as *const Tensor<B, 2>) };
                state_ref.clone()
            }
            3 => {
                // Sequences [batch, seq, features] -> flatten to [batch*seq, features]
                let state_ref: &Tensor<B, 3> = unsafe { &*(state as *const _ as *const Tensor<B, 3>) };
                let [batch, seq, features] = state_ref.dims();
                state_ref.clone().reshape([batch * seq, features])
            }
            _ => panic!("Unsupported state dimension: {}", STATE_DIM),
        };

        // Concatenate state and action: [batch, state_dim + action_dim]
        let input = Tensor::cat(vec![state_2d, action.clone()], 1);

        // Forward through MLP
        self.mlp.forward(input)
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        self.mlp.soft_update(&other.mlp, tau)
    }
}

/// Helper struct to use two MLPs as SAC Actor (mean and log_std networks)
///
/// The actor outputs Gaussian policy parameters for continuous actions.
#[derive(Module, Debug)]
pub struct MLPActor<B: Backend> {
    shared: MLP<B>,
    mean_head: MLP<B>,
    log_std_head: MLP<B>,
}

impl<B: Backend> MLPActor<B> {
    /// Create a new MLP-based actor
    ///
    /// # Arguments
    /// - `state_dim` - Dimension of state
    /// - `action_dim` - Dimension of action
    /// - `hidden_layers` - Hidden layer sizes for shared trunk (e.g., [256, 256])
    /// - `device` - Device to create the network on
    pub fn new(state_dim: usize, action_dim: usize, hidden_layers: Vec<usize>, device: &B::Device) -> Self {
        use crate::nn::MLPConfig;

        let hidden_dim = *hidden_layers.last().unwrap_or(&256);

        // Shared trunk
        let shared = MLPConfig::new(state_dim, hidden_layers, hidden_dim).init(device);

        // Mean head (no hidden layers, just linear projection)
        let mean_head = MLPConfig::new(hidden_dim, vec![], action_dim).init(device);

        // Log std head (no hidden layers, just linear projection)
        let log_std_head = MLPConfig::new(hidden_dim, vec![], action_dim).init(device);

        Self { shared, mean_head, log_std_head }
    }
}

// D=1: Single state
impl<B: AutodiffBackend> SACActorModel<B, 1> for MLPActor<B> {
    fn forward(&self, state: &Tensor<B, 1>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Convert to batch format
        let batched = state.clone().unsqueeze_dim(0);

        // Process through shared trunk
        let features: Tensor<B, 2> = self.shared.forward(batched);

        // Get mean and log_std from separate heads
        let mean = self.mean_head.forward(features.clone());
        let log_std = self.log_std_head.forward(features);

        // Clamp log_std to reasonable range
        let log_std = log_std.clamp(-20.0, 2.0);

        (mean, log_std)
    }
}

// D=2: Batch of states
impl<B: AutodiffBackend> SACActorModel<B, 2> for MLPActor<B> {
    fn forward(&self, state: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Process through shared trunk
        let features: Tensor<B, 2> = self.shared.forward(state.clone());

        // Get mean and log_std from separate heads
        let mean = self.mean_head.forward(features.clone());
        let log_std = self.log_std_head.forward(features);

        // Clamp log_std to reasonable range
        let log_std = log_std.clamp(-20.0, 2.0);

        (mean, log_std)
    }
}

// D=3: Sequences
impl<B: AutodiffBackend> SACActorModel<B, 3> for MLPActor<B> {
    fn forward(&self, state: &Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, seq, features] = state.dims();

        // Reshape to [batch*seq, features]
        let reshaped = state.clone().reshape([batch * seq, features]);

        // Process through shared trunk
        let trunk_out: Tensor<B, 2> = self.shared.forward(reshaped);

        // Get mean and log_std from separate heads
        let mean = self.mean_head.forward(trunk_out.clone());
        let log_std = self.log_std_head.forward(trunk_out);

        // Clamp log_std to reasonable range
        let log_std = log_std.clamp(-20.0, 2.0);

        // Reshape back to [batch, seq*action_dim]
        let [_, action_dim] = mean.dims();
        let mean = mean.reshape([batch, seq * action_dim]);
        let log_std = log_std.reshape([batch, seq * action_dim]);

        (mean, log_std)
    }
}