//! Twin Delayed Deep Deterministic Policy Gradient (TD3)
//!
//! TD3 is an off-policy actor-critic algorithm for continuous action spaces.
//! It improves upon DDPG with three key innovations:
//! 1. Twin critic networks to reduce overestimation bias
//! 2. Delayed policy updates (update actor less frequently than critics)
//! 3. Target policy smoothing (add noise to target actions)
//!
//! # Algorithm Overview
//!
//! TD3 learns a deterministic policy π(s) that maximizes expected return in continuous
//! action environments. It maintains:
//! - **Actor network**: Outputs deterministic actions a = π(s)
//! - **Two critic networks**: Estimate Q-values Q₁(s,a) and Q₂(s,a)
//! - **Target networks**: Slowly-updated copies of actor and critics for stable learning
//!
//! ## Key Features
//! - **Clipped Double Q-learning**: Uses min(Q₁, Q₂) to reduce overestimation
//! - **Delayed policy updates**: Updates actor every N critic updates (default: 2)
//! - **Target policy smoothing**: Adds noise to target actions to regularize Q-function
//! - **Experience replay**: Learns from past experiences stored in replay buffer
//!
//! # Usage Example
//!
//! ## 1. Implementing Networks
//!
//! First, create actor and critic networks that implement the required traits:
//!
//! ```rust,ignore
//! use burn::{
//!     config::Config,
//!     module::Module,
//!     nn::{Linear, LinearConfig, Relu},
//!     tensor::backend::AutodiffBackend,
//!     prelude::*,
//! };
//! use rl::algo::td3::{TD3ActorModel, TD3CriticModel};
//!
//! // Actor network: outputs deterministic actions
//! #[derive(Module, Debug)]
//! pub struct ActorModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     action_head: Linear<B>,
//! }
//!
//! impl<B: AutodiffBackend> TD3ActorModel<B, 2> for ActorModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
//!         let x = Relu.forward(self.fc1.forward(state));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         self.action_head.forward(x).tanh() // Bounded actions in [-1, 1]
//!     }
//!
//!     fn soft_update(&mut self, other: &Self, tau: f32) {
//!         // Update: self = tau * other + (1 - tau) * self
//!         soft_update_linear(&mut self.fc1, &other.fc1, tau);
//!         soft_update_linear(&mut self.fc2, &other.fc2, tau);
//!         soft_update_linear(&mut self.action_head, &other.action_head, tau);
//!     }
//! }
//!
//! // Critic network: outputs Q-value for state-action pair
//! #[derive(Module, Debug)]
//! pub struct CriticModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     q_head: Linear<B>,
//! }
//!
//! impl<B: AutodiffBackend> TD3CriticModel<B, 2> for CriticModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
//!         // Concatenate state and action
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
//!     type Action = [f32; 1];  // torque in [-2, 2]
//!
//!     fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
//!         // Apply action, update state, return (next_state, reward)
//!         // Return None for terminal states
//!     }
//!
//!     fn reset(&mut self) -> Self::State {
//!         // Reset environment and return initial state
//!     }
//!
//!     fn random_action(&self) -> Self::Action {
//!         // Return random action for exploration
//!     }
//! }
//!
//! impl ContinuousActionSpace for PendulumEnv {
//!     fn action_dim(&self) -> usize { 1 }
//!     fn action_bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
//!         Some((vec![-2.0], vec![2.0]))  // [low, high]
//!     }
//! }
//! ```
//!
//! ## 3. Training the Agent
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, Wgpu};
//! use rl::algo::td3::{TD3Agent, TD3AgentConfig};
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
//!     let config = TD3AgentConfig {
//!         memory_capacity: 100_000,
//!         memory_batch_size: 256,
//!         gamma: 0.99,              // Discount factor
//!         tau: 0.005,               // Soft update coefficient
//!         lr_actor: 3e-4,           // Actor learning rate
//!         lr_critic: 3e-4,          // Critic learning rate
//!         policy_delay: 2,          // Update actor every 2 critic updates
//!         target_policy_noise: 0.2, // Std of target action noise
//!         target_noise_clip: 0.5,   // Clip target noise to [-0.5, 0.5]
//!         exploration_noise: 0.1,   // Std of exploration noise
//!         learning_starts: 1000,    // Start learning after N steps
//!         gradient_clip: Some(1.0), // Gradient clipping value
//!     };
//!
//!     let mut agent = TD3Agent::new(actor, critic1, critic2, &env, config, &device);
//!
//!     // Train for multiple episodes
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
//! - **Batch size**: 100-256
//! - **Learning rates**: 1e-4 to 3e-4
//! - **Discount (γ)**: 0.99
//! - **Soft update (τ)**: 0.005
//! - **Policy delay**: 2
//!
//! ## Noise Settings
//! - **Exploration noise**: 0.1 (Gaussian noise added during training)
//! - **Target policy noise**: 0.2 (noise added to target actions)
//! - **Noise clip**: 0.5 (limit noise magnitude)
//!
//! # Tips
//! - Start training after collecting enough experiences (e.g., 1000 steps)
//! - Use gradient clipping to prevent exploding gradients
//! - Scale actions to match environment bounds
//! - Monitor critic losses - they should decrease over time
//!
//! Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use std::collections::HashMap;

use crate::{
    env::{ContinuousActionSpace, Environment, Report},
    memory::{Exp, ReplayMemory},
    nn::MLP,
    traits::{ToTensor, TrainableAgent, TrainingMetrics},
};


/// TD3 Actor model trait - outputs deterministic actions
pub trait TD3ActorModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: state -> action
    /// States are batched with dimension D, actions are always 2D (batch, action_dim)
    ///
    /// OPTIMIZATION: Takes reference to avoid cloning tensors in forward passes
    fn forward(&self, state: &Tensor<B, D>) -> Tensor<B, 2>;

    /// Soft update: θ′ ← τθ + (1 − τ)θ′
    fn soft_update(&mut self, other: &Self, tau: f32);
}

/// TD3 Critic model trait - outputs Q-values for state-action pairs
pub trait TD3CriticModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: (state, action) -> Q-value
    /// States are batched with dimension D, actions are always 2D (batch, action_dim)
    ///
    /// OPTIMIZATION: Takes references to avoid cloning tensors in forward passes
    fn forward(&self, state: &Tensor<B, D>, action: &Tensor<B, 2>) -> Tensor<B, 2>;

    /// Soft update: θ′ ← τθ + (1 − τ)θ′
    fn soft_update(&mut self, other: &Self, tau: f32);
}

/// Configuration for TD3 agent
#[derive(Debug, Clone)]
pub struct TD3AgentConfig {
    /// Replay memory capacity (default: 100,000)
    pub memory_capacity: usize,
    /// Batch size for learning (default: 256)
    pub memory_batch_size: usize,
    /// Discount factor γ (default: 0.99)
    pub gamma: f32,
    /// Soft update coefficient τ (default: 0.005)
    pub tau: f32,
    /// Actor learning rate (default: 3e-4)
    pub lr_actor: f64,
    /// Critic learning rate (default: 3e-4)
    pub lr_critic: f64,
    /// Policy update delay - update actor every N critic updates (default: 2)
    pub policy_delay: usize,
    /// Target policy noise - std of noise added to target actions (default: 0.2)
    pub target_policy_noise: f32,
    /// Target policy noise clip - clip noise to [-c, c] (default: 0.5)
    pub target_noise_clip: f32,
    /// Exploration noise - std of noise added to actions during training (default: 0.1)
    pub exploration_noise: f32,
    /// Number of steps before starting learning (default: 1000)
    pub learning_starts: usize,
    /// Gradient clipping value (default: Some(1.0))
    pub gradient_clip: Option<f32>,
}

impl Default for TD3AgentConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 100_000,
            memory_batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            policy_delay: 2,
            target_policy_noise: 0.2,
            target_noise_clip: 0.5,
            exploration_noise: 0.1,
            learning_starts: 1000,
            gradient_clip: Some(1.0),
        }
    }
}

/// TD3 Agent for continuous action spaces
///
/// This agent is generic over:
/// - `B`: Autodiff backend (e.g., Wgpu, NdArray)
/// - `Actor`: Actor network implementing TD3ActorModel
/// - `Critic`: Critic network implementing TD3CriticModel
/// - `E`: Environment with continuous action space
/// - `STATE_DIM`: Dimension of state tensor
pub struct TD3Agent<B, Actor, Critic, E, const STATE_DIM: usize>
where
    B: AutodiffBackend,
    E: Environment + ContinuousActionSpace,
    Actor: AutodiffModule<B>,
    Critic: AutodiffModule<B>,
{
    // Networks (Option for ownership during optimization)
    actor: Option<Actor>,
    target_actor: Option<Actor>,
    critic1: Option<Critic>,
    critic2: Option<Critic>,
    target_critic1: Option<Critic>,
    target_critic2: Option<Critic>,

    // Memory
    memory: ReplayMemory<E>,

    // Device
    device: &'static B::Device,

    // Hyperparameters
    gamma: f32,
    tau: f32,
    lr_actor: f64,
    lr_critic: f64,
    policy_delay: usize,
    target_policy_noise: f32,
    target_noise_clip: f32,
    exploration_noise: f32,
    learning_starts: usize,
    gradient_clip: Option<f32>,

    // Training state
    total_steps: usize,
    update_count: usize,
    is_learning: bool,
    eval_mode: bool,
    action_dim: usize,
    action_bounds: Option<(Vec<f32>, Vec<f32>)>,

    // Optimizers (stored to avoid recreation)
    optimizer_actor: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Actor, B>,
    optimizer_critic1: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Critic, B>,
    optimizer_critic2: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Critic, B>,
}

impl<B, Actor, Critic, E, const STATE_DIM: usize> TD3Agent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: TD3ActorModel<B, STATE_DIM>,
    Critic: TD3CriticModel<B, STATE_DIM>,
    E: Environment + ContinuousActionSpace,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
    Vec<E::Action>: ToTensor<B, 2, Float>,
{
    /// Get the current size of the replay buffer
    pub fn buffer_size(&self) -> usize {
        self.memory.len()
    }

    /// Create a new TD3 agent
    pub fn new(
        actor: Actor,
        critic1: Critic,
        critic2: Critic,
        env: &E,
        config: TD3AgentConfig,
        device: &'static B::Device,
    ) -> Self {
        let target_actor = actor.clone();
        let target_critic1 = critic1.clone();
        let target_critic2 = critic2.clone();

        let action_dim = env.action_dim();
        let action_bounds = env.action_bounds();

        // Initialize optimizers once
        let optimizer_actor = if let Some(clip_val) = config.gradient_clip {
            burn::optim::AdamWConfig::new()
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(clip_val)))
                .init()
        } else {
            burn::optim::AdamWConfig::new().init()
        };

        let optimizer_critic1 = if let Some(clip_val) = config.gradient_clip {
            burn::optim::AdamWConfig::new()
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(clip_val)))
                .init()
        } else {
            burn::optim::AdamWConfig::new().init()
        };

        let optimizer_critic2 = if let Some(clip_val) = config.gradient_clip {
            burn::optim::AdamWConfig::new()
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(clip_val)))
                .init()
        } else {
            burn::optim::AdamWConfig::new().init()
        };

        Self {
            actor: Some(actor),
            target_actor: Some(target_actor),
            critic1: Some(critic1),
            critic2: Some(critic2),
            target_critic1: Some(target_critic1),
            target_critic2: Some(target_critic2),
            memory: ReplayMemory::new(config.memory_capacity, config.memory_batch_size),
            device,
            gamma: config.gamma,
            tau: config.tau,
            lr_actor: config.lr_actor,
            lr_critic: config.lr_critic,
            policy_delay: config.policy_delay,
            target_policy_noise: config.target_policy_noise,
            target_noise_clip: config.target_noise_clip,
            exploration_noise: config.exploration_noise,
            learning_starts: config.learning_starts,
            gradient_clip: config.gradient_clip,
            total_steps: 0,
            update_count: 0,
            is_learning: false,
            eval_mode: false,
            action_dim,
            action_bounds,
            optimizer_actor,
            optimizer_critic1,
            optimizer_critic2,
        }
    }

    /// Select an action using the actor network
    ///
    /// In eval mode: returns deterministic action from actor
    /// In train mode: adds Gaussian exploration noise
    pub fn act(&self, state: &E::State) -> E::Action {
        let state_vec = vec![state.clone()];
        let state_tensor = state_vec.to_tensor(self.device);

        let actor = self.actor.as_ref().unwrap();
        let action_tensor = actor.forward(&state_tensor);

        // Add exploration noise during training
        let action_tensor = if self.eval_mode {
            action_tensor
        } else {
            let noise = Tensor::random_like(
                &action_tensor,
                burn::tensor::Distribution::Normal(0.0, self.exploration_noise as f64),
            );
            action_tensor.add(noise).clamp(-1.0, 1.0)

        };

        // Convert tensor to action (assuming single action in batch)
        let action_data = action_tensor.to_data();
        let action_slice = action_data.as_slice::<B::FloatElem>().unwrap();

        // Scale action to environment bounds if specified
        let mut action_values: Vec<f32> = action_slice
            .iter()
            .map(|x| x.elem::<f32>())
            .collect();

        if let Some((ref low, ref high)) = self.action_bounds {
            for (i, val) in action_values.iter_mut().enumerate() {
                // Assumes actor outputs in [-1, 1] (e.g., using tanh)
                // Scale to [low, high]
                *val = low[i] + (*val + 1.0) * 0.5 * (high[i] - low[i]);
                *val = val.clamp(low[i], high[i]);
            }
        }

        // Convert Vec<f32> to E::Action
        // This assumes E::Action can be constructed from a slice
        // For 1D actions like [f32; 1], we need to construct it properly
        unsafe {
            std::ptr::read(action_values.as_ptr() as *const E::Action)
        }
    }

    /// Run one episode (placeholder - needs environment-specific action selection)
    pub fn go(&mut self, env: &mut E) -> Report {
        let mut state = env.reset();

        let mut episode_reward = 0.0;
        let mut steps = 0;

        while env.is_active() {
            let action = self.act(&state);

            let (next_state_opt, reward) = env.step(action.clone());

            episode_reward += reward;
            steps += 1;
            self.total_steps += 1;

            // Store experience
            self.memory.push(Exp {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state_opt.clone(),
            });

            // Learn if enough experiences collected (use is_learning flag for efficiency)
            if !self.is_learning && self.total_steps >= self.learning_starts {
                self.is_learning = true;
            }

            if self.is_learning {
                self.learn_internal();
            }

            if next_state_opt.is_none() {
                break;
            }
            state = next_state_opt.unwrap();
        }

        let mut report = Report::new(vec!["reward", "steps"]);
        report.entry("reward").and_modify(|x| *x = episode_reward as f64);
        report.entry("steps").and_modify(|x| *x = steps as f64);
        report
    }

    /// Update the agent (critic and optionally actor)
    fn learn_internal(&mut self) -> Option<TrainingMetrics> {
        let batch = match self.memory.sample_zipped() {
            Some(b) => b,
            None => return None,
        };

        // OPTIMIZATION: Create common tensors once and pass by reference
        // This avoids duplicate batch.states.clone().to_tensor() calls in update_critics and update_actor
        let batch_size = batch.states.len();
        let first_state = batch.states[0].clone();
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let rewards = batch.rewards.as_slice();
        let next_states  = batch.next_states;

        let critic1_loss_val;
        let critic2_loss_val;
        let actor_loss_val;

        // Update critics
        {
            let (c1_loss, c2_loss) = self.update_critics(first_state,batch_size,rewards, &states, &actions,&next_states);
            critic1_loss_val = c1_loss;
            critic2_loss_val = c2_loss;
        }

        // Delayed policy update
        if self.update_count % self.policy_delay == 0 {
            actor_loss_val = self.update_actor( &states);
            self.soft_update_targets();
        } else {
            actor_loss_val = 0.0;
        }

        self.update_count += 1;

        Some(TrainingMetrics {
            policy_loss: actor_loss_val,
            value_loss: (critic1_loss_val + critic2_loss_val) / 2.0,
            entropy: 0.0, // TD3 is deterministic
            approx_kl: None,
            clip_fraction: None,
            early_stopped: false,
            n_updates: 1,
            extra: {
                let mut map = HashMap::new();
                map.insert("critic1_loss".to_string(), critic1_loss_val);
                map.insert("critic2_loss".to_string(), critic2_loss_val);
                map
            },
        })
    }

    /// Update critic networks and return loss values
    /// OPTIMIZATION: Takes pre-created state and action tensors to avoid duplicate cloning
    fn update_critics(
        &mut self,
        first_state: <E as Environment>::State,
        batch_size: usize,
        rewards: &[f32],
        states: &Tensor<B, STATE_DIM>,
        actions: &Tensor<B, 2>,
        next_states: &Vec<Option<<E as Environment>::State>>,
    ) -> (f32, f32) {


        // Pre-allocate rewards tensor directly (avoid intermediate allocation)
        let rewards = Tensor::<B, 1>::from_data(
            TensorData::from(rewards).convert::<B::FloatElem>(),
            self.device,
        )
        .reshape([batch_size, 1]);

        // Build mask and next_states in a single pass to avoid double iteration
        let mut mask_vec = Vec::with_capacity(batch_size);
        let mut next_states_vec = Vec::with_capacity(batch_size);

        for ns in next_states.iter() {
            if let Some(ref state) = ns {
                mask_vec.push(1.0_f32);
                next_states_vec.push(state.to_owned());
            } else {
                mask_vec.push(0.0_f32);
                next_states_vec.push(first_state.clone());
            }
        }

        let not_done_mask = Tensor::<B, 1>::from_data(
            TensorData::from(mask_vec.as_slice()).convert::<B::FloatElem>(),
            self.device,
        )
        .reshape([batch_size, 1]);

        let next_states = next_states_vec.to_tensor(self.device);


        // Compute target

        let target_actor = self.target_actor.as_ref().unwrap();
        let next_actions = target_actor.forward(&next_states);

        // Add clipped noise to target actions
        let noise = Tensor::random_like(
            &next_actions,
            burn::tensor::Distribution::Normal(0.0, self.target_policy_noise as f64),
        )
        .clamp(-self.target_noise_clip, self.target_noise_clip);
        let next_actions_noisy = next_actions.add(noise).clamp(-1.0, 1.0);

        // Compute target Q-values (minimum of two critics)
        // OPTIMIZATION: Share next_states reference instead of cloning
        let target_critic1 = self.target_critic1.as_ref().unwrap();
        let target_critic2 = self.target_critic2.as_ref().unwrap();

        let target_q1 = target_critic1.forward(&next_states, &next_actions_noisy);
        let target_q2 = target_critic2.forward(&next_states, &next_actions_noisy);
        let target_q = target_q1.min_pair(target_q2);

        // Compute target: r + γ * (1 - done) * min_i Q_target_i(s', a')
        let target = rewards.add(
            target_q
                .mul(not_done_mask)
                .mul_scalar(self.gamma),
        )
        .detach();


        let loss1_val;
        let loss2_val;

        // Update critic 1
        {
            let critic1 = self.critic1.take().unwrap();

            let q1 = critic1.forward(states, actions);
            let loss1 = (q1 - target.clone()).powf_scalar(2.0).mean();
            loss1_val = loss1.clone().into_scalar().elem::<f32>();

            let grads1 = loss1.backward();

            let grads1 = GradientsParams::from_grads(grads1, &critic1);

            let updated = self.optimizer_critic1.step(self.lr_critic, critic1, grads1);

            self.critic1 = Some(updated);
        }

        // Update critic 2
        {
            let critic2 = self.critic2.take().unwrap();

            let q2 = critic2.forward(states, actions);
            let loss2 = (q2 - target).powf_scalar(2.0).mean();
            loss2_val = loss2.clone().into_scalar().elem::<f32>();

            let grads2 = loss2.backward();

            let grads2 = GradientsParams::from_grads(grads2, &critic2);

            let updated = self.optimizer_critic2.step(self.lr_critic, critic2, grads2);

            self.critic2 = Some(updated);
        }

        (loss1_val, loss2_val)
    }

    /// Update actor network and return loss value
    /// OPTIMIZATION: Takes pre-created state tensor to avoid duplicate cloning
    fn update_actor(&mut self, states: &Tensor<B, STATE_DIM>) -> f32 {

        let actor = self.actor.take().unwrap();



        let actions = actor.forward(states);
        let critic1 = self.critic1.as_ref().unwrap();
        let q_values = critic1.forward(states, &actions);

        // Maximize Q-value = minimize -Q
        let loss = q_values.neg().mean();
        let loss_val = loss.clone().into_scalar().elem::<f32>();


        let grads = loss.backward();

        let grads = GradientsParams::from_grads(grads, &actor);

        self.actor = Some(self.optimizer_actor.step(self.lr_actor, actor, grads));

        loss_val
    }

    /// Soft update target networks
    #[inline]
    fn soft_update_targets(&mut self) {
        let actor = self.actor.as_ref().unwrap();
        let critic1 = self.critic1.as_ref().unwrap();
        let critic2 = self.critic2.as_ref().unwrap();

        self.target_actor.as_mut().unwrap().soft_update(actor, self.tau);
        self.target_critic1.as_mut().unwrap().soft_update(critic1, self.tau);
        self.target_critic2.as_mut().unwrap().soft_update(critic2, self.tau);
    }
}

/// Implementation of TrainableAgent trait for TD3Agent
impl<B, Actor, Critic, E, const STATE_DIM: usize> TrainableAgent<E>
    for TD3Agent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: TD3ActorModel<B, STATE_DIM>,
    Critic: TD3CriticModel<B, STATE_DIM>,
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

        // Store experience in replay memory only during training
        if !self.eval_mode {
            self.memory.push(Exp {
                state,
                action,
                reward,
                next_state: next_state_opt.clone(),
            });
        }

        self.total_steps += 1;

        // Check if we should start learning
        if !self.is_learning && self.total_steps >= self.learning_starts && !self.eval_mode {
            self.is_learning = true;
        }

        // Learn if conditions are met
        let metrics = if self.should_learn() {
            self.learn_internal()
        } else {
            None
        };

        (reward, done, metrics)
    }

    fn should_learn(&self) -> bool {
        self.is_learning && !self.eval_mode
    }

    fn learn(&mut self, _next_state: &E::State, _done: bool) -> TrainingMetrics {
        // TD3 doesn't need next_state or done for learn() since it uses replay memory
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
        // TD3 uses replay memory, so no per-episode reset needed
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
// MLP Implementations for TD3
// ================================================================================================

// TD3 Actor: MLP with tanh output activation for bounded continuous actions
// D=1: Single state
impl<B: AutodiffBackend> TD3ActorModel<B, 1> for MLP<B> {
    fn forward(&self, state: &Tensor<B, 1>) -> Tensor<B, 2> {
        let batched = state.clone().unsqueeze_dim(0);
        MLP::forward_tanh(self, batched)
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}

// D=2: Batch of states
impl<B: AutodiffBackend> TD3ActorModel<B, 2> for MLP<B> {
    fn forward(&self, state: &Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward_tanh(self, state.clone())
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}

// D=3: Sequences
impl<B: AutodiffBackend> TD3ActorModel<B, 3> for MLP<B> {
    fn forward(&self, state: &Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq, features] = state.dims();
        let reshaped = state.clone().reshape([batch * seq, features]);
        let output: Tensor<B, 2> = MLP::forward_tanh(self, reshaped);
        let [_, actions] = output.dims();
        output.reshape([batch, seq * actions])
    }

    fn soft_update(&mut self, other: &Self, tau: f32) {
        MLP::soft_update(self, other, tau)
    }
}

// TD3 Critic: Similar to SAC critic - concatenates state and action
use burn::module::Module;

/// Helper struct to use MLP as TD3 Critic
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

impl<B: AutodiffBackend, const STATE_DIM: usize> TD3CriticModel<B, STATE_DIM> for MLPCritic<B> {
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
