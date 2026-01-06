//! Proximal Policy Optimization (PPO) implementation
//!
//! PPO is an on-policy actor-critic algorithm that improves upon A2C by using a clipped
//! surrogate objective to prevent excessively large policy updates.
//!
//! # Key Features
//!
//! - **Clipped objective**: Prevents policy from changing too much in one update
//! - **Multiple epochs**: Reuses trajectory data for multiple gradient steps
//! - **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates
//! - **Value clipping**: Optional clipping of value function updates
//!
//! # Algorithm Overview
//!
//! 1. Collect trajectories using current policy
//! 2. Compute advantages using GAE
//! 3. For K epochs:
//!    - Sample mini-batches from trajectories
//!    - Update actor with clipped PPO objective
//!    - Update critic with value loss (optionally clipped)
//!
//! # Usage Example
//!
//! ```ignore
//! use rl::algo::ppo::{PPOAgent, PPOAgentConfig};
//!
//! let config = PPOAgentConfig {
//!     gamma: 0.99,
//!     gae_lambda: 0.95,
//!     clip_epsilon: 0.2,
//!     n_steps: 2048,
//!     n_epochs: 10,
//!     batch_size: 64,
//!     ..Default::default()
//! };
//!
//! let mut agent = PPOAgent::new(actor, critic, config, &device);
//! agent.go(&mut env);
//! ```
//!
//! # Hyperparameters
//!
//! - `clip_epsilon`: Clipping parameter (typically 0.1-0.3)
//! - `gae_lambda`: GAE lambda for advantage estimation (0.9-0.99)
//! - `n_steps`: Steps to collect before update (1024-4096)
//! - `n_epochs`: Number of optimization epochs per update (3-10)
//! - `batch_size`: Mini-batch size for updates (32-256)
//!
//! # Tips
//!
//! - PPO is more stable than A2C but requires more computation
//! - Use larger n_steps for better sample efficiency
//! - Increase n_epochs if learning is slow, decrease if overfitting
//! - Monitor KL divergence and clip fraction for debugging
//!
//! Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

use std::usize;

use burn::{
    module::AutodiffModule, optim::{GradientsParams, Optimizer}, prelude::*, tensor::{BasicOps, backend::AutodiffBackend}
};
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    env::{Environment, Report},
    nn::MLP,
    traits::{ ToTensor, TrainableAgent, TrainingMetrics}
};


/// PPO Actor model trait for discrete actions
/// Outputs action probabilities
pub trait PPOActorModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: state -> action logits
    ///
    /// Takes states with dimension D and returns a 2D tensor of action logits.
    /// The exact shape of the output depends on the input dimension:
    /// - D=1 `[features]` → `[1, num_actions]`
    /// - D=2 `[batch, features]` → `[batch, num_actions]`
    /// - D=3 `[batch, seq, features]` → `[batch, seq*num_actions]` (flattened)
    ///
    /// The output is always 2D to maintain compatibility with PPO's training logic.
    fn forward(&self, state: Tensor<B, D>) -> Tensor<B, 2>;
}

/// PPO Critic model trait
/// Outputs state values V(s)
pub trait PPOCriticModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: state -> value
    ///
    /// Takes states with dimension D and returns a 2D tensor of state values.
    /// The exact shape of the output depends on the input dimension:
    /// - D=1 `[features]` → `[1, 1]`
    /// - D=2 `[batch, features]` → `[batch, 1]`
    /// - D=3 `[batch, seq, features]` → `[batch, seq]` (one value per timestep)
    ///
    /// The output is always 2D to maintain compatibility with PPO's training logic.
    fn forward(&self, state: Tensor<B, D>) -> Tensor<B, 2>;
}

/// Configuration for PPO agent
#[derive(Debug, Clone)]
pub struct PPOAgentConfig {
    /// Discount factor γ (default: 0.99)
    pub gamma: f32,
    /// GAE lambda λ for advantage estimation (default: 0.95)
    pub gae_lambda: f32,
    /// Clipping parameter ε (default: 0.2)
    pub clip_epsilon: f32,
    /// Actor learning rate (default: 3e-4)
    pub lr_actor: f64,
    /// Critic learning rate (default: 3e-4)
    pub lr_critic: f64,
    /// Entropy coefficient for exploration (default: 0.01)
    pub entropy_coef: f32,
    /// Value loss coefficient (default: 0.5)
    pub value_coef: f32,
    /// Number of steps to collect before update (default: 2048)
    pub n_steps: usize,
    /// Number of optimization epochs per update (default: 10)
    pub n_epochs: usize,
    /// Mini-batch size for updates (default: 64)
    pub batch_size: usize,
    /// Gradient clipping value (default: Some(0.5))
    pub gradient_clip: Option<f32>,
    /// Maximum KL divergence before early stopping (default: None)
    pub target_kl: Option<f32>,
    /// Clip value function loss (default: false)
    pub clip_value_loss: bool,
}

impl Default for PPOAgentConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            entropy_coef: 0.01,
            value_coef: 0.5,
            n_steps: 2048,
            n_epochs: 10,
            batch_size: 64,
            gradient_clip: Some(0.5),
            target_kl: None,
            clip_value_loss: false,
        }
    }
}

/// Trajectory storage for PPO
#[derive(Clone, Debug)]
struct Trajectory<S> {
    states: Vec<S>,
    action_indices: Vec<usize>,
    rewards: Vec<f32>,
    log_probs: Vec<f32>,
    values: Vec<f32>,
    dones: Vec<bool>,
}

impl<S> Trajectory<S> {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            action_indices: Vec::new(),
            rewards: Vec::new(),
            log_probs: Vec::new(),
            values: Vec::new(),
            dones: Vec::new(),
        }
    }

    fn push(&mut self, state: S, action_idx: usize, reward: f32, log_prob: f32, value: f32, done: bool) {
        self.states.push(state);
        self.action_indices.push(action_idx);
        self.rewards.push(reward);
        self.log_probs.push(log_prob);
        self.values.push(value);
        self.dones.push(done);
    }

    fn clear(&mut self) {
        self.states.clear();
        self.action_indices.clear();
        self.rewards.clear();
        self.log_probs.clear();
        self.values.clear();
        self.dones.clear();
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// PPO Agent for discrete action spaces
///
/// This agent is generic over:
/// - `B`: Autodiff backend (e.g., Wgpu, NdArray)
/// - `Actor`: Actor network implementing PPOActorModel
/// - `Critic`: Critic network implementing PPOCriticModel
/// - `E`: Environment with discrete action space
/// - `STATE_DIM`: Dimension of state tensor
pub struct PPOAgent<B, Actor, Critic, E, const D: usize>
where
    B: AutodiffBackend,
    E: Environment,
    Actor: AutodiffModule<B>,
    Critic: AutodiffModule<B>,
{
    // Networks (Option for ownership during optimization)
    actor: Option<Actor>,
    critic: Option<Critic>,

    // Trajectory buffer
    trajectory: Trajectory<E::State>,

    // Device
    device: &'static B::Device,

    // Hyperparameters
    gamma: f32,
    gae_lambda: f32,
    clip_epsilon: f32,
    lr_actor: f64,
    lr_critic: f64,
    entropy_coef: f32,
    value_coef: f32,
    n_steps: usize,
    n_epochs: usize,
    batch_size: usize,
    target_kl: Option<f32>,
    clip_value_loss: bool,

    // Training state
    total_steps: usize,
    learn_mode: bool,

    // Optimizers (stored to avoid recreation)
    optimizer_actor: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Actor, B>,
    optimizer_critic: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Critic, B>,
}

impl<B, Actor, Critic, E, const D: usize> PPOAgent<B, Actor, Critic, E, D>
where
    B: AutodiffBackend,
    Actor: PPOActorModel<B, D>,
    Critic: PPOCriticModel<B, D>,
    E: Environment,
    Vec<E::State>: ToTensor<B, D, Float>,
{
    /// Create a new PPO agent
    pub fn new(
        actor: Actor,
        critic: Critic,
        config: PPOAgentConfig,
        device: &'static B::Device,
    ) -> Self {
        // Initialize optimizers once
        let optimizer_actor = if let Some(clip_val) = config.gradient_clip {
            burn::optim::AdamWConfig::new()
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(clip_val)))
                .init()
        } else {
            burn::optim::AdamWConfig::new().init()
        };

        let optimizer_critic = if let Some(clip_val) = config.gradient_clip {
            burn::optim::AdamWConfig::new()
                .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(clip_val)))
                .init()
        } else {
            burn::optim::AdamWConfig::new().init()
        };

        Self {
            actor: Some(actor),
            critic: Some(critic),
            trajectory: Trajectory::new(),
            device,
            gamma: config.gamma,
            gae_lambda: config.gae_lambda,
            clip_epsilon: config.clip_epsilon,
            lr_actor: config.lr_actor,
            lr_critic: config.lr_critic,
            entropy_coef: config.entropy_coef,
            value_coef: config.value_coef,
            n_steps: config.n_steps,
            n_epochs: config.n_epochs,
            batch_size: config.batch_size,
            target_kl: config.target_kl,
            clip_value_loss: config.clip_value_loss,
            total_steps: 0,
            learn_mode: true,

            optimizer_actor,
            optimizer_critic,
        }
    }



    /// Get total number of environment steps taken
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Select an action for a given state
    ///
    /// Returns (action_index, log_prob, value_estimate)
    pub fn select_action(&self, state: &E::State) -> (usize, f32, f32)
    where
        E::Action: From<usize>,
    {
        let state_tensor = vec![state.clone()].to_tensor(self.device);
        let actor = self.actor.as_ref().unwrap();
        let critic = self.critic.as_ref().unwrap();

        let logits = actor.forward(state_tensor.clone());
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);

        // Get value estimate
        let value = critic.forward(state_tensor);
        let value_scalar = value.to_data().as_slice::<f32>().unwrap()[0];

        // Sample action from categorical distribution
        let probs_data = probs.to_data();
        let probs_slice = probs_data.as_slice::<f32>().unwrap();

        use rand::{distributions::Distribution, distributions::WeightedIndex, thread_rng};
        let dist = WeightedIndex::new(probs_slice).unwrap();
        let action_idx = dist.sample(&mut thread_rng());

        // Get log probability of selected action
        let log_prob_data = log_probs.to_data();
        let log_prob_slice = log_prob_data.as_slice::<f32>().unwrap();
        let log_prob = log_prob_slice[action_idx];

        (action_idx, log_prob, value_scalar)
    }

    /// Store a transition in the trajectory
    pub fn store_transition(
        &mut self,
        state: E::State,
        action_idx: usize,
        reward: f32,
        log_prob: f32,
        value: f32,
        done: bool,
    ) {
        self.trajectory.push(state, action_idx, reward, log_prob, value, done);
        self.total_steps += 1;
    }

    /// Check if the agent should train (trajectory buffer is full or episode ended)
    pub fn should_train(&self) -> bool {
        self.trajectory.len() >= self.n_steps && self.learn_mode
    }

    /// Train the agent and return metrics
    ///
    /// Call this when `should_train()` returns true
    pub fn train(&mut self, next_state: &E::State, done: bool) -> Option<TrainingMetrics> {
        self.learn(next_state, done)
    }

    /// Run one episode
    pub fn go(&mut self, env: &mut E) -> Report
    where
        E::Action: From<usize>,
    {
        env.reset();

        let mut episode_reward = 0.0;
        let mut steps = 0;

        while env.is_active() {
            // Select action using actor network

            let (step_reward, done, _) = self.step(env);
            episode_reward += step_reward;
            steps += 1;
            self.total_steps += 1;

            if done {
                break;
            }
        }

        let mut report = Report::new(vec!["reward", "steps"]);
        report.entry("reward").and_modify(|x| *x = episode_reward as f64);
        report.entry("steps").and_modify(|x| *x = steps as f64);
        report
    }

    /// Update the agent using collected trajectory
    fn learn(&mut self, next_state: &E::State, done: bool) -> Option<TrainingMetrics> {
        if self.trajectory.is_empty() {
            return None;
        }

        // Initialize metrics accumulator
        let mut total_policy_loss = 0.0_f32;
        let mut total_value_loss = 0.0_f32;
        let mut total_entropy = 0.0_f32;
        let mut total_approx_kl = 0.0_f32;
        let mut total_clip_fraction = 0.0_f32;
        let mut n_updates = 0;

        // Compute advantages and returns using GAE
        let (returns, advantages) = self.compute_gae(next_state, done);


        // Pre-allocate action indices Vec
        let action_indices_i32: Vec<i32> = self.trajectory
            .action_indices
            .iter()
            .map(|&x| x as i32)
            .collect();

        let action_indices = Tensor::<B, 1, Int>::from_data(
            TensorData::from(action_indices_i32.as_slice()).convert::<B::IntElem>(),
            self.device,
        )
        .unsqueeze_dim::<2>(1);

        let old_log_probs = Tensor::<B, 1>::from_data(
            TensorData::from(self.trajectory.log_probs.as_slice()).convert::<B::FloatElem>(),
            self.device,
        );


        // Create mini-batch indices
        let n_samples = self.trajectory.len();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Multiple epochs of optimization
        for _epoch in 0..self.n_epochs {

            // Shuffle indices
            indices.shuffle(&mut thread_rng());

            // Process mini-batches
            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // Create batch tensors by gathering
                let batch_states = self.gather_states(batch_indices);
                let batch_action_indices = self.gather_tensor(&action_indices, batch_indices);
                let batch_old_log_probs = self.gather_tensor(&old_log_probs, batch_indices);
                let batch_returns = self.gather_tensor(&returns, batch_indices);
                let batch_advantages = self.gather_tensor(&advantages, batch_indices);

                // Update actor
                {
                    let actor = self.actor.take().unwrap();
                    let logits = actor.forward(batch_states.clone());
                    let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);

                    // Gather log probs for taken actions
                    let action_log_probs = log_probs.clone().gather(1, batch_action_indices.clone()).squeeze_dims(&[1]);

                    // Compute ratio and clipped ratio
                    let ratio = (action_log_probs.clone() - batch_old_log_probs.clone()).exp();
                    let clipped_ratio = ratio.clone().clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);

                    // PPO clipped objective
                    let surr1 = ratio.clone() * batch_advantages.clone();
                    let surr2 = clipped_ratio * batch_advantages.clone();
                    // Use min: element-wise minimum of surr1 and surr2
                    let policy_loss = surr1.clone().min_pair(surr2).mean().neg();

                    // Entropy bonus
                    let probs = burn::tensor::activation::softmax(logits, 1);
                    let entropy = (probs.clone() * log_probs).sum_dim(1).neg().mean();
                    let entropy_loss = entropy.clone() * self.entropy_coef;

                    let actor_loss = policy_loss.clone() - entropy_loss;

                    // Extract metrics before backward pass
                    let policy_loss_val = policy_loss.clone().into_scalar().elem::<f32>();
                    let entropy_val = entropy.clone().into_scalar().elem::<f32>();

                    // Calculate approximate KL divergence: E[log(ratio)] ≈ E[ratio - 1]
                    let approx_kl_val = (batch_old_log_probs - action_log_probs).mean().into_scalar().elem::<f32>();

                    // Calculate clip fraction: fraction of ratios outside [1-eps, 1+eps]
                    let lower_bound = Tensor::from_floats([1.0 - self.clip_epsilon], self.device);
                    let upper_bound = Tensor::from_floats([1.0 + self.clip_epsilon], self.device);
                    let clip_lower = ratio.clone().lower(lower_bound);
                    let clip_upper = ratio.clone().greater(upper_bound);
                    let clipped = clip_lower.int() + clip_upper.int();
                    let clip_frac = clipped.float().mean().into_scalar().elem::<f32>();

                    // Accumulate metrics
                    total_policy_loss += policy_loss_val;
                    total_entropy += entropy_val;
                    total_approx_kl += approx_kl_val;
                    total_clip_fraction += clip_frac;

                    let actor_grads = actor_loss.backward();

                    let actor_grads = GradientsParams::from_grads(actor_grads, &actor);

                    self.actor = Some(self.optimizer_actor.step(self.lr_actor, actor, actor_grads));
                }

                // Update critic
                {
                    let critic = self.critic.take().unwrap();
                    let values = critic.forward(batch_states).squeeze_dims(&[1]);

                    let critic_loss = if self.clip_value_loss {
                        // Value clipping (optional, helps stability)
                        let batch_old_values = self.gather_values(batch_indices);
                        let values_clipped = batch_old_values.clone() +
                            (values.clone() - batch_old_values.clone()).clamp(-self.clip_epsilon, self.clip_epsilon);
                        let loss1 = (batch_returns.clone() - values).powf_scalar(2.0);
                        let loss2 = (batch_returns - values_clipped).powf_scalar(2.0);
                        loss1.clone().max_pair(loss2).mean() * self.value_coef
                    } else {
                        (batch_returns - values).powf_scalar(2.0).mean() * self.value_coef
                    };

                    // Extract value loss metric before backward pass
                    let value_loss_val = critic_loss.clone().into_scalar().elem::<f32>();
                    total_value_loss += value_loss_val;

                    // Increment update counter
                    n_updates += 1;

                    let critic_grads = critic_loss.backward();

                    let critic_grads = GradientsParams::from_grads(critic_grads, &critic);

                    self.critic = Some(self.optimizer_critic.step(self.lr_critic, critic, critic_grads));
                }
            }

            // Early stopping based on KL divergence (optional)
            if let Some(_target_kl) = self.target_kl {
                // TODO: Implement KL divergence tracking
            }
        }

        // Clear trajectory
        self.trajectory.clear();

        // Compute average metrics
        let metrics = TrainingMetrics {
            policy_loss: if n_updates > 0 { total_policy_loss / n_updates as f32 } else { 0.0 },
            value_loss: if n_updates > 0 { total_value_loss / n_updates as f32 } else { 0.0 },
            entropy: if n_updates > 0 { total_entropy / n_updates as f32 } else { 0.0 },
            approx_kl: Some(if n_updates > 0 { total_approx_kl / n_updates as f32 } else { 0.0 }),
            clip_fraction: Some(if n_updates > 0 { total_clip_fraction / n_updates as f32 } else { 0.0 }),
            n_updates,
            early_stopped: false,
            extra: std::collections::HashMap::new(),
        };

    
        Some(metrics)
    }

    /// Compute Generalized Advantage Estimation (GAE)
    #[inline]
    fn compute_gae(&self, next_state: &E::State, done: bool) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let n = self.trajectory.len();

        // Compute bootstrap value
        let bootstrap_value = if done {
            0.0
        } else {
            let next_state_tensor = vec![next_state.clone()].to_tensor(self.device);
            let critic = self.critic.as_ref().unwrap();
            let value = critic.forward(next_state_tensor);
            value.to_data().as_slice::<f32>().unwrap()[0]
        };

        // Compute advantages using GAE
        let mut advantages = vec![0.0; n];
        let mut gae = 0.0;

        for i in (0..n).rev() {
            let next_value = if i == n - 1 {
                bootstrap_value
            } else {
                self.trajectory.values[i + 1]
            };

            let delta = self.trajectory.rewards[i] + self.gamma * next_value * (1.0 - self.trajectory.dones[i] as u8 as f32) - self.trajectory.values[i];
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.trajectory.dones[i] as u8 as f32) * gae;
            advantages[i] = gae;
        }

        // Compute returns
        let returns: Vec<f32> = advantages.iter()
            .zip(self.trajectory.values.iter())
            .map(|(adv, val)| adv + val)
            .collect();

        let returns_tensor = Tensor::<B, 1>::from_data(
            TensorData::from(returns.as_slice()).convert::<B::FloatElem>(),
            self.device,
        );

        let advantages_tensor = Tensor::<B, 1>::from_data(
            TensorData::from(advantages.as_slice()).convert::<B::FloatElem>(),
            self.device,
        );

        // Normalize advantages
        let adv_mean = advantages_tensor.clone().mean();
        let adv_std = advantages_tensor.clone().sub(adv_mean.clone()).powf_scalar(2.0).mean().sqrt();
        let advantages_normalized = (advantages_tensor - adv_mean) / (adv_std + 1e-8);

        (returns_tensor, advantages_normalized)
    }

    /// Helper to gather states by indices
    #[inline]
    fn gather_states(&self, indices: &[usize]) -> Tensor<B, D, Float> {
        let batch_states: Vec<E::State> = indices.iter()
            .map(|&i| self.trajectory.states[i].clone())
            .collect();
        batch_states.to_tensor(self.device)
    }

    /// Helper to gather tensor values by indices
    #[inline]
    fn gather_tensor<const DIM: usize, K: BasicOps<B>>(&self, tensor: &Tensor<B, DIM, K>, indices: &[usize]) -> Tensor<B, DIM, K> {
        let indices_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::from(indices.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice()).convert::<B::IntElem>(),
            self.device,
        );
        tensor.clone().select(0, indices_tensor)
    }

    /// Helper to gather values by indices
    #[inline]
    fn gather_values(&self, indices: &[usize]) -> Tensor<B, 1> {
        let batch_values: Vec<f32> = indices.iter()
            .map(|&i| self.trajectory.values[i])
            .collect();
        Tensor::<B, 1>::from_data(
            TensorData::from(batch_values.as_slice()).convert::<B::FloatElem>(),
            self.device,
        )
    }
}


/// Implementation of TrainableAgent trait for PPO
///
/// This provides a PyTorch-like training API with step(), train(), and evaluate()
impl<B, Actor, Critic, E, const STATE_DIM: usize> TrainableAgent<E> for PPOAgent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: PPOActorModel<B, STATE_DIM>,
    Critic: PPOCriticModel<B, STATE_DIM>,
    E: Environment,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
    E::Action: From<usize>,
{
    type StepInfo = (f32, bool,Option<TrainingMetrics>); // (reward, done)

    /// Take one step in the environment (like PyTorch forward pass)
    ///
    /// Collects experience but doesn't train. Call train() when should_train() returns true.
    fn step(&mut self, env: &mut E) -> Self::StepInfo {

        let state = env.current_state();

        // Select action
        let (action_idx, log_prob, value) = self.select_action(&state);

        // Take action in environment
        let (next_state_opt, reward) = env.step(action_idx.into());
        let done: bool = next_state_opt.is_none();

        // Store transition
        self.trajectory.push(state.clone(), action_idx, reward, log_prob, value, done);
        self.total_steps += 1;

        // Update state


        if self.should_train() || done {
            if let Some(state) = next_state_opt {
                let metrics = self.train(&state, done);
                return (reward, done, metrics)
            }
        }
        (reward, done, None)

    }

    fn eval(&mut self) {
        self.learn_mode = false;
    }

    /// Make agent in training mode
    fn train(&mut self) {
        self.learn_mode = true;
    }

    /// Check if agent should train (buffer full)
    ///
    /// PPO trains when trajectory buffer reaches n_steps, not at episode boundaries.
    /// Multiple episodes can contribute to a single training batch.
    fn should_learn(&self) -> bool {
        self.trajectory.len() >= self.n_steps && self.learn_mode
    }

    /// Train on collected experience (like PyTorch backward + optimizer.step)
    ///
    /// Returns training metrics (losses, entropy, KL, etc.)
    fn learn(&mut self, next_state: &E::State, done: bool) -> TrainingMetrics {


        // Call internal learn function
        self.learn(&next_state, done).unwrap_or_default()
    }

   

    /// Reset episode state (clear trajectory, keep weights)
    fn reset_episode(&mut self) {
        self.trajectory.clear();
    }

    /// Get total environment steps
    fn total_steps(&self) -> usize {
        self.total_steps
    }
}

// ============================================================================
// MLP Implementations for PPO
// ============================================================================

/// Implementation for 2D tensors (batch of states)
/// `[batch, features]` → `[batch, actions]`
impl<B: AutodiffBackend> PPOActorModel<B, 2> for MLP<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward(self, state)
    }
}


/// Implementation for 2D tensors (batch of states)
/// `[batch, features]` → `[batch, 1]`
impl<B: AutodiffBackend> PPOCriticModel<B, 2> for MLP<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward(self, state)
    }
}
