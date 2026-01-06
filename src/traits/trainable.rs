//! Trainable agent trait for fine-grained training control
//!
//! This trait provides a unified API for training RL agents with detailed metrics
//! and step-by-step control, enabling:
//! - Monitoring training progress (losses, entropy, KL divergence, etc.)
//! - Validation during training to detect overfitting
//! - Early stopping based on validation performance
//! - Custom training loops with full control

use crate::env::Environment;

/// Training metrics returned after each training update
///
/// These metrics allow monitoring the training progress and detecting issues
/// like overfitting, divergence, or lack of exploration.
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    /// Policy/actor loss (lower is better for most algorithms)
    pub policy_loss: f32,

    /// Value/critic loss (lower is better)
    pub value_loss: f32,

    /// Entropy of the policy (higher means more exploration)
    pub entropy: f32,

    /// Approximate KL divergence from old policy (for PPO/TRPO)
    pub approx_kl: Option<f32>,

    /// Fraction of probability ratios that were clipped (for PPO)
    pub clip_fraction: Option<f32>,

    /// Number of gradient updates performed
    pub n_updates: usize,

    /// Whether training was stopped early (e.g., due to KL threshold)
    pub early_stopped: bool,

    /// Additional algorithm-specific metrics
    pub extra: std::collections::HashMap<String, f32>,
}

/// Trait for trainable RL agents with fine-grained control
///
/// This trait provides methods for:
/// - Collecting experience without training
/// - Training on collected experience with detailed metrics
/// - Evaluating the agent's performance
/// - Accessing internal state for debugging
///
/// # Example
///
/// ```ignore
/// // Collect experience
/// let step_info = agent.step(&mut env);
///
/// // Train when ready
/// if agent.should_train() {
///     let metrics = agent.train();
///     println!("Policy loss: {}", metrics.policy_loss);
///
///     // Validate on separate episodes
///     let eval_metrics = agent.evaluate(&mut eval_env, 10);
///     println!("Mean reward: {}", eval_metrics.mean_reward);
/// }
/// ```
pub trait TrainableAgent<E: Environment> {
    /// Information returned after each environment step
    type StepInfo;

    /// Take one step in the environment
    ///
    /// This collects experience but does not train the agent.
    /// Returns information about the step (reward, done, etc.)
    fn step(&mut self, env: &mut E) -> Self::StepInfo;

    /// Check if the agent is ready to train
    ///
    /// Returns true when enough experience has been collected
    /// (e.g., trajectory buffer is full for PPO, replay buffer has enough samples for DQN)
    /// if agent is in evaluate mode this should return false
    fn should_learn(&self) -> bool;

    /// Train the agent on collected experience
    ///
    /// Returns detailed metrics about the training update.
    /// This should be called after `should_train()` returns true.
    fn learn(&mut self, next_state: &E::State, done: bool) -> TrainingMetrics;


    /// Reset the agent's episode state
    ///
    /// Clears trajectory buffers and resets counters, but keeps learned weights.
    fn reset_episode(&mut self);

    /// Get total number of environment steps taken
    fn total_steps(&self) -> usize;

    // Make agent in evaluation mode
    fn eval(&mut self);

    // Make agent in training mode
    fn train(&mut self);
}
