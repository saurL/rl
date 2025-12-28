use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rl::algo::sac::{SACActorModel, SACCriticModel};

/// Actor network configuration
#[derive(Config, Debug)]
pub struct ActorConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    #[config(default = 64)]
    pub hidden_size: usize,
}

/// Actor network for SAC
///
/// Architecture: state → fc1 (hidden) → ReLU → fc2 (hidden) → ReLU →
///               mean_head (action_dim), log_std_head (action_dim)
#[derive(Module, Debug)]
pub struct ActorModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mean_head: Linear<B>,
    log_std_head: Linear<B>,
}

impl ActorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorModel<B> {
        ActorModel {
            fc1: LinearConfig::new(self.state_dim, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            mean_head: LinearConfig::new(self.hidden_size, self.action_dim).init(device),
            log_std_head: LinearConfig::new(self.hidden_size, self.action_dim).init(device),
        }
    }
}

impl<B: AutodiffBackend> SACActorModel<B, 2> for ActorModel<B> {
    fn forward(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = Relu.forward(self.fc1.forward(state));
        let x = Relu.forward(self.fc2.forward(x));

        let mean = self.mean_head.forward(x.clone());
        let log_std = self.log_std_head.forward(x);

        (mean, log_std)
    }

    fn sample_action(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mean, log_std) = self.forward(state);

        // Clamp log_std for numerical stability
        let log_std = log_std.clamp(-20.0, 2.0);
        let std = log_std.clone().exp();

        // Sample from Gaussian: action = mean + std * ε, where ε ~ N(0,1)
        let epsilon = Tensor::random_like(&mean, burn::tensor::Distribution::Normal(0.0, 1.0));
        let action_unbounded = mean.clone().add(std.clone().mul(epsilon));

        // Apply tanh squashing for bounded actions
        let action = action_unbounded.clone().tanh();

        // Compute log probability with correction for tanh squashing
        // log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))

        // Gaussian log probability: -0.5 * ((x - μ) / σ)² - log(σ) - 0.5*log(2π)
        let normalized_diff = action_unbounded.clone().sub(mean.clone()).div(std.clone());
        let log_prob_gaussian = normalized_diff
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .sub(log_std.clone())
            .sub_scalar(0.5 * (2.0 * std::f64::consts::PI).ln() as f32);

        // Sum over action dimensions
        let log_prob_unbounded = log_prob_gaussian.sum_dim(1);

        // Tanh correction: log(1 - tanh²(x))
        let tanh_correction = action
            .clone()
            .powf_scalar(2.0)
            .mul_scalar(-1.0)
            .add_scalar(1.0)
            .add_scalar(1e-6) // For numerical stability
            .log()
            .sum_dim(1);

        let log_prob = log_prob_unbounded.sub(tanh_correction).unsqueeze_dim(1);

        // Mean action for deterministic evaluation
        let mean_action = mean.tanh();

        (action, log_prob, mean_action)
    }
}

/// Critic network configuration
#[derive(Config, Debug)]
pub struct CriticConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    #[config(default = 256)]
    pub hidden_size: usize,
}

/// Critic network for SAC
///
/// Architecture: concat(state, action) → fc1 (hidden) → ReLU →
///               fc2 (hidden) → ReLU → fc3 (1)
#[derive(Module, Debug)]
pub struct CriticModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CriticModel<B> {
        let input_dim = self.state_dim + self.action_dim;
        CriticModel {
            fc1: LinearConfig::new(input_dim, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            fc3: LinearConfig::new(self.hidden_size, 1).init(device),
        }
    }
}

impl<B: AutodiffBackend> SACCriticModel<B, 2> for CriticModel<B> {
    fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate state and action
        let x = Tensor::cat(vec![state, action], 1);

        let x = Relu.forward(self.fc1.forward(x));
        let x = Relu.forward(self.fc2.forward(x));
        self.fc3.forward(x)
    }

    fn soft_update(self, other: &Self, tau: f32) -> Self {
        Self {
            fc1: soft_update_linear(self.fc1, &other.fc1, tau),
            fc2: soft_update_linear(self.fc2, &other.fc2, tau),
            fc3: soft_update_linear(self.fc3, &other.fc3, tau),
        }
    }
}

/// Helper function for soft updating tensor parameters
///
/// θ′ ← τθ + (1 − τ)θ′
fn soft_update_tensor<B: Backend, const D: usize>(
    this: burn::module::Param<Tensor<B, D>>,
    that: &burn::module::Param<Tensor<B, D>>,
    tau: f32,
) -> burn::module::Param<Tensor<B, D>> {
    this.map(|tensor| tensor * (1.0 - tau) + that.val() * tau)
}

/// Helper function for soft updating Linear layers
///
/// θ′ ← τθ + (1 − τ)θ′
fn soft_update_linear<B: Backend>(
    mut this: Linear<B>,
    that: &Linear<B>,
    tau: f32,
) -> Linear<B> {
    this.weight = soft_update_tensor(this.weight, &that.weight, tau);
    this.bias = match (this.bias, &that.bias) {
        (Some(b1), Some(b2)) => Some(soft_update_tensor(b1, b2, tau)),
        _ => None,
    };

    this
}
