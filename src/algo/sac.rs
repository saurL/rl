use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};

use crate::{
    env::{ContinuousActionSpace, Environment},
    memory::{Exp, Memory, PrioritizedReplayMemory, ReplayMemory},
    traits::{BoolToTensor, ToTensor},
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
    fn forward(&self, state: Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// Sample an action from the policy and compute log probability
    ///
    /// This handles the tanh squashing for bounded actions.
    ///
    /// Returns: (action, log_prob, mean_action)
    /// - action: [batch_size, action_dim] - squashed to [-1, 1]
    /// - log_prob: [batch_size, 1] - log probability of the action
    /// - mean_action: [batch_size, action_dim] - mean squashed with tanh (for deterministic evaluation)
    fn sample_action(&self, state: Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);
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
    fn forward(&self, state: Tensor<B, STATE_DIM>, action: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Soft update the parameters of the target network
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    ///
    /// ```ignore
    /// target_critic = target_critic.soft_update(&critic, tau);
    /// ```
    fn soft_update(self, other: &Self, tau: f32) -> Self;
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

    // Action space info
    action_dim: usize,
    action_bounds: Option<(Vec<f32>, Vec<f32>)>,
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
            action_dim,
            action_bounds,
        }
    }

    /// Select an action given a state
    ///
    /// # Arguments
    /// - `state` - The current state
    /// - `deterministic` - If true, use mean action (for evaluation); if false, sample (for training)
    ///
    /// # Returns
    /// The selected action
    fn act(&self, state: E::State, deterministic: bool) -> E::Action {
        let state_tensor = vec![state].to_tensor(self.device);
        let actor = self.actor.as_ref().unwrap();

        let action_tensor = if deterministic {
            // Use mean action for deterministic evaluation
            let (_action, _log_prob, mean_action) = actor.sample_action(state_tensor);
            mean_action
        } else {
            // Sample action for exploration
            let (action, _log_prob, _mean_action) = actor.sample_action(state_tensor);
            action
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
        // This is a bit tricky - we need a way to convert from Vec<f32> to the action type
        // For now, we'll use a workaround assuming action is array-like
        // In a real implementation, we'd need a trait for this conversion
        unsafe {
            let ptr = scaled_action.as_ptr() as *const E::Action;
            ptr.read()
        }
    }

    /// Soft update target critic networks
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    fn soft_update_targets(&mut self) {
        let critic1 = self.critic1.as_ref().unwrap();
        let critic2 = self.critic2.as_ref().unwrap();
        let target_critic1 = self.target_critic1.take().unwrap();
        let target_critic2 = self.target_critic2.take().unwrap();

        self.target_critic1 = Some(target_critic1.soft_update(critic1, self.tau));
        self.target_critic2 = Some(target_critic2.soft_update(critic2, self.tau));
    }

    /// Update critic networks with clipped double-Q learning
    ///
    /// # Returns
    /// TD errors for each sample (for prioritized replay)
    fn update_critics(
        &mut self,
        batch: &crate::memory::ExpBatch<E>,
        weights: Option<&Tensor<B, 1>>,
        optimizer_critic1: &mut impl Optimizer<Critic, B>,
        optimizer_critic2: &mut impl Optimizer<Critic, B>,
    ) -> Vec<f32> {
        let batch_size = batch.states.len();

        // Convert batch to tensors
        let states = batch.states.clone().to_tensor(self.device);
        let actions = batch.actions.clone().to_tensor(self.device);
        let rewards = Tensor::<B, 1, Float>::from_data(
            TensorData::from(batch.rewards.as_slice()).convert::<B::FloatElem>(),
            self.device,
        )
        .unsqueeze_dim(1);

        // Create terminal mask
        let non_terminal_mask = batch
            .next_states
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .to_bool_tensor(self.device)
            .unsqueeze_dim(1);

        // Extract non-terminal next states
        let next_states = batch
            .next_states
            .iter()
            .filter_map(|s| s.as_ref().cloned())
            .collect::<Vec<_>>()
            .to_tensor(self.device);

        // Take ownership of networks
        let critic1 = self.critic1.take().unwrap();
        let critic2 = self.critic2.take().unwrap();
        let target_critic1 = self.target_critic1.as_ref().unwrap();
        let target_critic2 = self.target_critic2.as_ref().unwrap();
        let actor = self.actor.as_ref().unwrap();

        // Sample next actions from current policy
        let (next_actions, next_log_probs, _) = actor.sample_action(next_states.clone());

        // Compute target Q-values using clipped double-Q
        let target_q1 = target_critic1
            .forward(next_states.clone(), next_actions.clone())
            .detach();
        let target_q2 = target_critic2.forward(next_states, next_actions).detach();
        let target_q = target_q1.min_pair(target_q2); // Clipped double-Q

        // Apply entropy regularization: Q_target = r + γ(min(Q1', Q2') - α*log_π)
        let alpha_tensor = Tensor::<B, 1>::from_data(
            TensorData::from([self.alpha]).convert::<B::FloatElem>(),
            self.device,
        );
        let entropy_term = next_log_probs.mul_scalar(self.alpha);
        let target_value = target_q.sub(entropy_term);

        // Apply non-terminal mask
        let zeros = Tensor::zeros([batch_size, 1], self.device);
        let masked_target = zeros.mask_where(non_terminal_mask, target_value);

        // Compute TD target
        let td_target = rewards.add(masked_target.mul_scalar(self.gamma));

        // Compute current Q-values
        let q1_pred = critic1.forward(states.clone(), actions.clone());
        let q2_pred = critic2.forward(states.clone(), actions.clone());

        // Compute TD errors for PER
        let td_error1 = q1_pred.clone().sub(td_target.clone());
        let td_error2 = q2_pred.clone().sub(td_target.clone());
        let avg_td_errors: Tensor<B, 1> = td_error1
            .clone()
            .abs()
            .add(td_error2.clone().abs())
            .div_scalar(2.0)
            .squeeze();

        // Compute losses
        let loss1 = if let Some(w) = weights {
            let squared_errors: Tensor<B, 1> = td_error1.powf_scalar(2.0).squeeze();
            w.clone().mul(squared_errors).mean()
        } else {
            MseLoss::new().forward(q1_pred, td_target.clone(), Reduction::Mean)
        };

        let loss2 = if let Some(w) = weights {
            let squared_errors: Tensor<B, 1> = td_error2.powf_scalar(2.0).squeeze();
            w.clone().mul(squared_errors).mean()
        } else {
            MseLoss::new().forward(q2_pred, td_target, Reduction::Mean)
        };

        // Backprop and update both critics
        let grads1 = GradientsParams::from_grads(loss1.backward(), &critic1);
        let grads2 = GradientsParams::from_grads(loss2.backward(), &critic2);

        self.critic1 = Some(optimizer_critic1.step(self.lr_critic, critic1, grads1));
        self.critic2 = Some(optimizer_critic2.step(self.lr_critic, critic2, grads2));

        // Return TD errors for PER
        avg_td_errors.into_data().iter::<f32>().collect()
    }

    /// Update actor network
    ///
    /// Maximize Q - α*log_π
    fn update_actor(
        &mut self,
        batch: &crate::memory::ExpBatch<E>,
        optimizer_actor: &mut impl Optimizer<Actor, B>,
    ) {
        let states = batch.states.clone().to_tensor(self.device);

        let actor = self.actor.take().unwrap();
        let critic1 = self.critic1.as_ref().unwrap();
        let critic2 = self.critic2.as_ref().unwrap();

        // Sample actions from current policy
        let (actions, log_probs, _) = actor.sample_action(states.clone());

        // Compute Q-values for sampled actions (use min for conservative estimate)
        let q1 = critic1.forward(states.clone(), actions.clone());
        let q2 = critic2.forward(states, actions);
        let q_min = q1.min_pair(q2);

        // Actor loss: maximize Q - α*log_π ⟺ minimize α*log_π - Q
        let actor_loss = log_probs.mul_scalar(self.alpha).sub(q_min).mean();

        // Backprop and update actor
        let grads = GradientsParams::from_grads(actor_loss.backward(), &actor);
        self.actor = Some(optimizer_actor.step(self.lr_actor, actor, grads));
    }

    /// Update alpha (temperature) via automatic tuning
    ///
    /// Maximize entropy towards target_entropy
    fn update_alpha(
        &mut self,
        batch: &crate::memory::ExpBatch<E>,
        optimizer_alpha: &mut impl Optimizer<Tensor<B, 1>, B>,
    ) {
        if !self.auto_alpha {
            return;
        }

        let states = batch.states.clone().to_tensor(self.device);
        let actor = self.actor.as_ref().unwrap();

        let log_alpha = self.log_alpha.take().unwrap();

        // Sample actions to compute entropy
        let (_, log_probs, _) = actor.sample_action(states);

        // Alpha loss: -α * (log_π + target_entropy)
        // This maximizes entropy towards target_entropy
        let alpha_exp = log_alpha.clone().exp();
        let entropy_diff: Tensor<B, 1> = log_probs.add_scalar(self.target_entropy).squeeze();
        let alpha_loss = alpha_exp.mul(entropy_diff.detach()).neg().mean();

        // Update log_alpha
        let grads = GradientsParams::from_grads(alpha_loss.backward(), &log_alpha);
        let new_log_alpha = optimizer_alpha.step(self.lr_alpha, log_alpha, grads);

        // Update alpha value
        let alpha_scalar: B::FloatElem = new_log_alpha.clone().exp().into_scalar();
        self.alpha = alpha_scalar.elem();
        self.log_alpha = Some(new_log_alpha);
    }

    /// Main learning loop
    fn learn(
        &mut self,
        optimizer_actor: &mut impl Optimizer<Actor, B>,
        optimizer_critic1: &mut impl Optimizer<Critic, B>,
        optimizer_critic2: &mut impl Optimizer<Critic, B>,
        optimizer_alpha: &mut Option<impl Optimizer<Tensor<B, 1>, B>>,
    ) {
        if self.total_steps < self.learning_starts {
            return;
        }

        for _ in 0..self.gradient_steps {
            // Sample batch (with or without PER)
            let (batch, weights, indices) = match &mut self.memory {
                Memory::Base(memory) => {
                    let batch = memory.sample_zipped();
                    if let Some(b) = batch {
                        (b, None, None)
                    } else {
                        return;
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
                        return;
                    }
                }
            };

            // 1. Update critics (with clipped double-Q)
            let td_errors = self.update_critics(
                &batch,
                weights.as_ref(),
                optimizer_critic1,
                optimizer_critic2,
            );

            // 2. Update actor (delayed policy updates)
            if self.update_count % self.actor_update_interval == 0 {
                self.update_actor(&batch, optimizer_actor);

                // 3. Update alpha (if auto-tuning)
                if let Some(opt_alpha) = optimizer_alpha {
                    self.update_alpha(&batch, opt_alpha);
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
    }

    /// Train the agent on an environment
    ///
    /// This is the main training loop that collects experiences and learns from them.
    ///
    /// # Arguments
    /// - `env` - The environment to train on
    pub fn go(&mut self, env: &mut E) {
        // Create optimizers
        let mut optimizer_actor = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let mut optimizer_critic1 = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let mut optimizer_critic2 = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
            .init();
        let mut optimizer_alpha = if self.auto_alpha {
            Some(AdamWConfig::new().init())
        } else {
            None
        };

        let mut next_state = Some(env.reset());

        // Training loop
        while let Some(state) = next_state {
            // Select action (explore during training)
            let action = self.act(state.clone(), false);

            // Take step in environment
            let (next, reward) = env.step(action.clone());
            next_state = next.clone();

            // Create experience
            let exp = Exp {
                state,
                action,
                reward,
                next_state: next,
            };

            // Store in memory
            match &mut self.memory {
                Memory::Base(memory) => memory.push(exp),
                Memory::Prioritized(memory) => memory.push(exp),
            }

            // Learn
            self.learn(
                &mut optimizer_actor,
                &mut optimizer_critic1,
                &mut optimizer_critic2,
                &mut optimizer_alpha,
            );

            self.total_steps += 1;

            // Check if episode ended
            if next_state.is_none() {
                self.episodes_elapsed += 1;
                next_state = Some(env.reset());
            }
        }
    }
}
