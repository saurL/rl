//! Advantage Actor-Critic (A2C)
//!
//! A2C is an on-policy actor-critic algorithm that works with discrete action spaces.
//! It uses the advantage function A(s,a) = Q(s,a) - V(s) to reduce variance in policy gradients.
//!
//! # Algorithm Overview
//!
//! A2C combines value-based and policy-based methods by maintaining:
//! - **Actor network**: Outputs action probabilities π(a|s)
//! - **Critic network**: Estimates state value function V(s)
//!
//! The algorithm uses n-step returns and advantage estimation to update both networks
//! in an on-policy manner (learns from current policy interactions).
//!
//! ## Key Features
//! - **Advantage estimation**: Reduces variance by using A(s,a) = R(s,a) - V(s)
//! - **Entropy regularization**: Encourages exploration by maximizing policy entropy
//! - **N-step returns**: Uses multi-step bootstrapping for more accurate value estimates
//! - **Synchronous updates**: Updates after every n steps (simpler than async A3C)
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
//! use rl::algo::a2c::{A2CActorModel, A2CCriticModel};
//!
//! // Actor network: outputs action logits for discrete actions
//! #[derive(Module, Debug)]
//! pub struct ActorModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     action_head: Linear<B>,  // Output dimension = number of actions
//! }
//!
//! impl<B: AutodiffBackend> A2CActorModel<B, 2> for ActorModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
//!         let x = Relu.forward(self.fc1.forward(state));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         self.action_head.forward(x)  // Return logits (pre-softmax)
//!     }
//! }
//!
//! // Critic network: outputs state value V(s)
//! #[derive(Module, Debug)]
//! pub struct CriticModel<B: Backend> {
//!     fc1: Linear<B>,
//!     fc2: Linear<B>,
//!     value_head: Linear<B>,  // Output dimension = 1
//! }
//!
//! impl<B: AutodiffBackend> A2CCriticModel<B, 2> for CriticModel<B> {
//!     fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
//!         let x = Relu.forward(self.fc1.forward(state));
//!         let x = Relu.forward(self.fc2.forward(x));
//!         self.value_head.forward(x)  // Output shape: [batch, 1]
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
//! enum CartPoleAction {
//!     Left = 0,
//!     Right = 1,
//! }
//!
//! struct CartPoleEnv {
//!     state: [f32; 4],  // [x, x_dot, theta, theta_dot]
//!     // ... other fields
//! }
//!
//! impl Environment for CartPoleEnv {
//!     type State = [f32; 4];
//!     type Action = CartPoleAction;
//!
//!     fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
//!         // Apply action, update state, return (next_state, reward)
//!         // Return None when episode ends
//!     }
//!
//!     fn reset(&mut self) -> Self::State {
//!         // Reset environment and return initial state
//!     }
//!
//!     fn random_action(&self) -> Self::Action {
//!         // Return random action
//!     }
//! }
//!
//! impl DiscreteActionSpace for CartPoleEnv {
//!     fn actions(&self) -> Vec<Self::Action> {
//!         vec![CartPoleAction::Left, CartPoleAction::Right]
//!     }
//! }
//! ```
//!
//! ## 3. Training the Agent
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, Wgpu};
//! use rl::algo::a2c::{A2CAgent, A2CAgentConfig};
//!
//! type Backend = Autodiff<Wgpu>;
//!
//! fn main() {
//!     let device = Default::default();
//!     let mut env = CartPoleEnv::new();
//!
//!     // Create networks
//!     let state_dim = 4;
//!     let num_actions = 2;
//!     let actor = ActorConfig::new(state_dim, num_actions).init::<Backend>(&device);
//!     let critic = CriticConfig::new(state_dim).init::<Backend>(&device);
//!
//!     // Configure agent
//!     let config = A2CAgentConfig {
//!         gamma: 0.99,              // Discount factor
//!         lr_actor: 7e-4,           // Actor learning rate
//!         lr_critic: 7e-4,          // Critic learning rate
//!         entropy_coef: 0.01,       // Entropy regularization coefficient
//!         value_coef: 0.5,          // Value loss coefficient
//!         n_steps: 5,               // Number of steps before update
//!         gradient_clip: Some(0.5), // Gradient clipping value
//!     };
//!
//!     let mut agent = A2CAgent::new(actor, critic, config, &device);
//!
//!     // Train for multiple episodes
//!     for episode in 0..1000 {
//!         let report = agent.go(&mut env);
//!         println!("Episode {}: reward = {:.2}", episode, report.get("reward").unwrap());
//!     }
//! }
//! ```
//!
//! # Hyperparameter Tuning
//!
//! ## Common Settings
//! - **Learning rates**: 1e-4 to 1e-3 (often higher than off-policy methods)
//! - **Discount (γ)**: 0.99
//! - **N-steps**: 5-20 (balance between bias and variance)
//! - **Entropy coefficient**: 0.001-0.01 (higher = more exploration)
//! - **Value coefficient**: 0.5 (weight of critic loss vs actor loss)
//!
//! ## Network Architecture
//! - **Hidden layers**: 1-2 layers with 64-256 units
//! - **Activation**: ReLU or Tanh
//! - **Shared layers**: Can share early layers between actor and critic
//!
//! # Tips
//! - Use gradient clipping to prevent exploding gradients (common in on-policy methods)
//! - Adjust entropy coefficient based on exploration needs
//! - Monitor both actor and critic losses
//! - A2C is sample-efficient but can be unstable - consider PPO for more stability
//! - Use baseline subtraction (advantage) to reduce variance in policy gradients
//!
//! Reference: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::{distributions::WeightedIndex, thread_rng};

use crate::{
    env::{Environment, Report},
    nn::MLP,
    traits::{ToTensor, TrainableAgent},
};


/// A2C Actor model trait for discrete actions
/// Outputs action probabilities
pub trait A2CActorModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: state -> action logits
    /// States are batched with dimension D, outputs logits with shape (batch, num_actions)
    fn forward(&self, state: Tensor<B, D>) -> Tensor<B, 2>;
}

/// A2C Critic model trait
/// Outputs state values V(s)
pub trait A2CCriticModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass: state -> value
    /// States are batched with dimension D, outputs values with shape (batch, 1)
    fn forward(&self, state: Tensor<B, D>) -> Tensor<B, 2>;
}

/// Configuration for A2C agent
#[derive(Debug, Clone)]
pub struct A2CAgentConfig {
    /// Discount factor γ (default: 0.99)
    pub gamma: f32,
    /// Actor learning rate (default: 7e-4)
    pub lr_actor: f64,
    /// Critic learning rate (default: 7e-4)
    pub lr_critic: f64,
    /// Entropy coefficient for exploration (default: 0.01)
    pub entropy_coef: f32,
    /// Value loss coefficient (default: 0.5)
    pub value_coef: f32,
    /// Number of steps to collect before update (default: 5)
    pub n_steps: usize,
    /// Gradient clipping value (default: Some(0.5))
    pub gradient_clip: Option<f32>,
}

impl Default for A2CAgentConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lr_actor: 7e-4,
            lr_critic: 7e-4,
            entropy_coef: 0.01,
            value_coef: 0.5,
            n_steps: 5,
            gradient_clip: Some(0.5),
        }
    }
}

/// Trajectory storage for n-step returns
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

/// A2C Agent for discrete action spaces
///
/// This agent is generic over:
/// - `B`: Autodiff backend (e.g., Wgpu, NdArray)
/// - `Actor`: Actor network implementing A2CActorModel
/// - `Critic`: Critic network implementing A2CCriticModel
/// - `E`: Environment with discrete action space
/// - `STATE_DIM`: Dimension of state tensor
pub struct A2CAgent<B, Actor, Critic, E, const STATE_DIM: usize>
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
    lr_actor: f64,
    lr_critic: f64,
    entropy_coef: f32,
    value_coef: f32,
    n_steps: usize,
    gradient_clip: Option<f32>,

    // Training state
    total_steps: usize,
    learn_mode: bool,

    // Optimizers (stored to avoid recreation)
    optimizer_actor: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Actor, B>,
    optimizer_critic: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, Critic, B>,
}

impl<B, Actor, Critic, E, const STATE_DIM: usize> A2CAgent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: A2CActorModel<B, STATE_DIM>,
    Critic: A2CCriticModel<B, STATE_DIM>,
    E: Environment,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
{
    /// Create a new A2C agent
    pub fn new(
        actor: Actor,
        critic: Critic,
        config: A2CAgentConfig,
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
            lr_actor: config.lr_actor,
            lr_critic: config.lr_critic,
            entropy_coef: config.entropy_coef,
            value_coef: config.value_coef,
            n_steps: config.n_steps,
            gradient_clip: config.gradient_clip,
            total_steps: 0,
            learn_mode: true,
            optimizer_actor,
            optimizer_critic,
        }
    }

    /// Select an action for a given state
    ///
    /// Returns (action_index, log_prob, value_estimate)
    pub fn select_action(&self, state: &E::State) -> (usize, f32, f32) {
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
    pub fn train_step(&mut self, next_state: &E::State, done: bool) -> Option<crate::traits::TrainingMetrics> {
        self.learn_internal(next_state, done)
    }

    /// Run one episode (legacy method, prefer using TrainableAgent trait)
    pub fn go(&mut self, env: &mut E) -> Report
    where
        E::Action: From<usize>,
    {
        env.reset();

        let mut episode_reward = 0.0;
        let mut steps = 0;

        loop {
            let (reward, done, _) = self.step(env);
            episode_reward += reward;
            steps += 1;

            if done {
                break;
            }
        }

        let mut report = Report::new(vec!["reward", "steps"]);
        report.entry("reward").and_modify(|x| *x = episode_reward as f64);
        report.entry("steps").and_modify(|x| *x = steps as f64);
        report
    }

    /// Update the agent using collected trajectory (internal method)
    fn learn_internal(&mut self, next_state: &E::State, done: bool) -> Option<crate::traits::TrainingMetrics> {
        if self.trajectory.is_empty() {
            return None;
        }

    

        let states = self.trajectory.states.clone().to_tensor(self.device);

        // Pre-allocate action indices Vec to avoid reallocation
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


        // Compute returns and advantages
        let (returns, advantages) = self.compute_returns_and_advantages(next_state, done);

        // Update actor
        let (policy_loss_val, entropy_val) = {
            let actor = self.actor.take().unwrap();

            let logits = actor.forward(states.clone());

            // Use activation functions from burn
            let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);

            // Gather log probs for taken actions
            let action_log_probs = log_probs.clone().gather(1, action_indices).squeeze_dims(&[1]);

            // Compute entropy for exploration bonus
            let probs = burn::tensor::activation::softmax(logits, 1);
            let entropy = (probs * log_probs).sum_dim(1).neg().mean();

            // Actor loss: -advantages * log_prob - entropy_bonus
            let actor_loss =
                (advantages.clone().detach() * action_log_probs.neg()).mean() - entropy.clone() * self.entropy_coef;

            // Extract metrics before backward pass
            let policy_loss_val = actor_loss.clone().into_scalar().elem::<f32>();
            let entropy_val = entropy.clone().into_scalar().elem::<f32>();

            let actor_grads = actor_loss.backward();

            let actor_grads = GradientsParams::from_grads(actor_grads, &actor);

            self.actor = Some(self.optimizer_actor.step(self.lr_actor, actor, actor_grads));

            (policy_loss_val, entropy_val)
        };

        // Update critic
        let value_loss_val = {
            let critic = self.critic.take().unwrap();

            let values = critic.forward(states).squeeze_dims(&[1]);
            let critic_loss = (returns - values).powf_scalar(2.0).mean() * self.value_coef;

            // Extract value loss metric before backward pass
            let value_loss_val = critic_loss.clone().into_scalar().elem::<f32>();

            let critic_grads = critic_loss.backward();

            let critic_grads = GradientsParams::from_grads(critic_grads, &critic);

            self.critic = Some(self.optimizer_critic.step(self.lr_critic, critic, critic_grads));

            value_loss_val
        };

        // Clear trajectory
        self.trajectory.clear();

        // Return training metrics
        Some(crate::traits::TrainingMetrics {
            policy_loss: policy_loss_val,
            value_loss: value_loss_val,
            entropy: entropy_val,
            approx_kl: None,
            clip_fraction: None,
            n_updates: 1,
            early_stopped: false,
            extra: std::collections::HashMap::new(),
        })
    }

    /// Compute returns and advantages
    #[inline]
    fn compute_returns_and_advantages(
    &self,
    next_state: &E::State,
    done: bool,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let n = self.trajectory.len();

    let bootstrap_value = if done {
        0.0
    } else {
        let next_state_tensor = vec![next_state.clone()].to_tensor(self.device);
        let critic = self.critic.as_ref().unwrap();
        let value = critic.forward(next_state_tensor);
        value.to_data().as_slice::<f32>().unwrap()[0]
    };

    let mut returns = vec![0.0; n];
    let mut running_return = bootstrap_value;

    for i in (0..n).rev() {
        if self.trajectory.dones[i] {
            running_return = 0.0;
        }
        running_return = self.trajectory.rewards[i] + self.gamma * running_return;
        returns[i] = running_return;
    }

    let states = self.trajectory.states.clone().to_tensor(self.device);
    let critic = self.critic.as_ref().unwrap();
    let values = critic.forward(states).squeeze_dims(&[1]);

    let returns_tensor = Tensor::<B, 1>::from_data(
        TensorData::from(returns.as_slice()).convert::<B::FloatElem>(),
        self.device,
    );

    let advantages = returns_tensor.clone() - values;
    let adv_mean = advantages.clone().mean();
    let adv_std = advantages.clone().var(0).sqrt() + 1e-8;
    let advantages_normalized = (advantages - adv_mean) / adv_std;

    (returns_tensor, advantages_normalized)
}

}
    
/// Implementation of TrainableAgent trait for A2C
///
/// This provides a PyTorch-like training API with step(), train(), and evaluate()
impl<B, Actor, Critic, E, const STATE_DIM: usize> crate::traits::TrainableAgent<E>
    for A2CAgent<B, Actor, Critic, E, STATE_DIM>
where
    B: AutodiffBackend,
    Actor: A2CActorModel<B, STATE_DIM>,
    Critic: A2CCriticModel<B, STATE_DIM>,
    E: Environment,
    Vec<E::State>: ToTensor<B, STATE_DIM, Float>,
    E::Action: From<usize>,
{
    type StepInfo = (f32, bool, Option<crate::traits::TrainingMetrics>); // (reward, done, metrics)

    /// Take one step in the environment
    ///
    /// Collects experience and trains when buffer is full
    fn step(&mut self, env: &mut E) -> Self::StepInfo {
        let state = env.current_state();

        // Select action
        let (action_idx, log_prob, value) = self.select_action(&state);

        // Take action in environment
        let (next_state_opt, reward) = env.step(action_idx.into());
        let done = next_state_opt.is_none();

        // Store transition
        self.trajectory.push(state.clone(), action_idx, reward, log_prob, value, done);
        self.total_steps += 1;

        // Train when buffer is full or episode ends
        if self.should_learn() || done {
            if let Some(next_state) = next_state_opt {
                let metrics = self.learn(&next_state, done);
                return (reward, done, Some(metrics));
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
    fn should_learn(&self) -> bool {
        self.trajectory.len() >= self.n_steps && self.learn_mode
    }

    /// Train on collected experience
    ///
    /// Returns training metrics (losses, entropy)
    fn learn(&mut self, next_state: &E::State, done: bool) -> crate::traits::TrainingMetrics {
        self.learn_internal(&next_state, done).unwrap_or_default()
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
// MLP Implementations for A2C
// ============================================================================

/// Implementation for 1D tensors (single state)
/// `[features]` → `[1, actions]`
impl<B: AutodiffBackend> A2CActorModel<B, 1> for MLP<B> {
    fn forward(&self, state: Tensor<B, 1>) -> Tensor<B, 2> {
        let batched = state.unsqueeze_dim(0); // [features] → [1, features]
        MLP::forward(self, batched) // [1, features] → [1, actions]
    }
}

/// Implementation for 2D tensors (batch of states)
/// `[batch, features]` → `[batch, actions]`
impl<B: AutodiffBackend> A2CActorModel<B, 2> for MLP<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward(self, state)
    }
}

/// Implementation for 3D tensors (sequences or multi-entity)
/// `[batch, seq/entities, features]` → `[batch*seq, actions]` then reshape to `[batch, seq*actions]`
impl<B: AutodiffBackend> A2CActorModel<B, 3> for MLP<B> {
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
}

/// Implementation for 1D tensors (single state)
/// `[features]` → `[1, 1]`
impl<B: AutodiffBackend> A2CCriticModel<B, 1> for MLP<B> {
    fn forward(&self, state: Tensor<B, 1>) -> Tensor<B, 2> {
        let batched = state.unsqueeze_dim(0); // [features] → [1, features]
        MLP::forward(self, batched) // [1, features] → [1, 1]
    }
}

/// Implementation for 2D tensors (batch of states)
/// `[batch, features]` → `[batch, 1]`
impl<B: AutodiffBackend> A2CCriticModel<B, 2> for MLP<B> {
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        MLP::forward(self, state)
    }
}

/// Implementation for 3D tensors (sequences or multi-entity)
/// `[batch, seq/entities, features]` → `[batch*seq, 1]` then reshape to `[batch, seq]`
impl<B: AutodiffBackend> A2CCriticModel<B, 3> for MLP<B> {
    fn forward(&self, state: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, seq, features] = state.dims();

        // Flatten: [batch, seq, features] → [batch*seq, features]
        let reshaped = state.reshape([batch * seq, features]);

        // Forward: [batch*seq, features] → [batch*seq, 1]
        let output = MLP::forward::<2>(self, reshaped);

        // Reshape: [batch*seq, 1] → [batch, seq]
        output.reshape([batch, seq])
    }
}
