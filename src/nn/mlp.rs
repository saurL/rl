/// Multi-Layer Perceptron (MLP) - Generic feedforward neural network
///
/// This module provides a flexible MLP implementation that can be used
/// for Actor and Critic networks in RL algorithms.

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{activation::relu, backend::Backend},
};

/// Configuration for Multi-Layer Perceptron
#[derive(Config, Debug)]
pub struct MLPConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions (e.g., [128, 128] for two hidden layers of 128 units each)
    pub hidden_layers: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Use ReLU activation for hidden layers (default: true)
    #[config(default = "true")]
    pub use_relu: bool,
    /// Use Tanh activation for output layer (default: false)
    #[config(default = "false")]
    pub use_tanh_output: bool,
}

/// Multi-Layer Perceptron implementation
///
/// Hidden layers use ReLU activation by default
/// Output layer has no activation by default (linear)
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl MLPConfig {
    /// Initialize the MLP with the given configuration
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let mut layers = Vec::new();

        if self.hidden_layers.is_empty() {
            // Direct input → output connection (no hidden layers)
            layers.push(LinearConfig::new(self.input_dim, self.output_dim).init(device));
        } else {
            // Input → First hidden layer
            layers.push(
                LinearConfig::new(self.input_dim, self.hidden_layers[0]).init(device),
            );

            // Hidden → Hidden connections
            for i in 0..self.hidden_layers.len() - 1 {
                layers.push(
                    LinearConfig::new(self.hidden_layers[i], self.hidden_layers[i + 1])
                        .init(device),
                );
            }

            // Last hidden → Output
            let last_hidden = *self.hidden_layers.last().unwrap();
            layers.push(LinearConfig::new(last_hidden, self.output_dim).init(device));
        }

        MLP { layers }
    }
}

impl<B: Backend> MLP<B> {
    /// Generic forward pass - works with any tensor dimension
    ///
    /// Applies ReLU activation to all hidden layers, no activation on output layer.
    ///
    /// Works with:
    /// - D=1: Single example `[features]`
    /// - D=2: Batch processing `[batch, features]` (most common)
    /// - D=3: Sequences or multi-entity `[batch, sequence/entities, features]`
    /// - D=4+: Higher dimensional data (e.g., spatial-temporal)
    ///
    /// The last dimension is always treated as the feature dimension.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut x: Tensor<B, D> = input;

        // All layers except the last one (with ReLU activation)
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(x);
            x = relu(x);
        }

        // Last layer (no activation)
        x = self.layers.last().unwrap().forward(x);

        x
    }

    /// Generic forward pass with tanh output activation
    ///
    /// Useful for continuous action spaces bounded in [-1, 1].
    /// Works with any tensor dimension (see `forward` for details).
    pub fn forward_tanh<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = input;

        // All layers except the last one (with ReLU activation)
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(x);
            x = relu(x);
        }

        // Last layer (with tanh activation)
        x = self.layers.last().unwrap().forward(x);
        x = x.tanh();

        x
    }

    /// Soft update: θ′ ← τθ + (1 − τ)θ′
    ///
    /// Used for target networks in DQN, SAC, TD3.
    /// Updates `self` (target) toward `other` (policy) by factor `tau`.
    pub fn soft_update(&mut self, other: &Self, tau: f32) {
        for (target_layer, policy_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            soft_update_linear_inplace(target_layer, policy_layer, tau);
        }
    }
}

// Helper functions for soft updates
fn soft_update_tensor_inplace<B: Backend, const D: usize>(
    this: &mut Param<Tensor<B, D>>,
    that: &Param<Tensor<B, D>>,
    tau: f32,
) {
    // CRITICAL FIX: Use .detach() to prevent gradient graph accumulation
    // Without detach(), the autodiff graph grows indefinitely, causing exponential slowdown
    *this = this.clone().map(|tensor| tensor * (1.0 - tau) + that.val().detach() * tau);
}

fn soft_update_linear_inplace<B: Backend>(
    this: &mut Linear<B>,
    that: &Linear<B>,
    tau: f32,
) {
    soft_update_tensor_inplace(&mut this.weight, &that.weight, tau);

    match (&mut this.bias, &that.bias) {
        (Some(b1), Some(b2)) => soft_update_tensor_inplace(b1, b2, tau),
        _ => {},
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_mlp_forward_1d() {
        let device = NdArrayDevice::default();

        // Create a simple MLP: 4 → [64, 64] → 2
        let config = MLPConfig::new(4, vec![64, 64], 2);
        let mlp = config.init::<NdArray>(&device);

        // Single state: [features]
        let input = Tensor::<NdArray, 1>::random(
            [4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let output: Tensor<NdArray, 1> = mlp.forward(input);

        // Check output shape: [output_dim]
        assert_eq!(output.shape().dims, [2]);
    }

    #[test]
    fn test_mlp_forward_2d() {
        let device = NdArrayDevice::default();

        // Create a simple MLP: 4 → [64, 64] → 2
        let config = MLPConfig::new(4, vec![64, 64], 2);
        let mlp = config.init::<NdArray>(&device);

        // Batch of 8 states: [batch, features]
        let input = Tensor::<NdArray, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let output: Tensor<NdArray, 2> = mlp.forward(input);

        // Check output shape: [batch, output_dim]
        assert_eq!(output.shape().dims, [8, 2]);
    }

    #[test]
    fn test_mlp_forward_3d() {
        let device = NdArrayDevice::default();

        // Create a simple MLP: 4 → [64, 64] → 2
        let config = MLPConfig::new(4, vec![64, 64], 2);
        let mlp = config.init::<NdArray>(&device);

        // Sequences: [batch, sequence_length, features]
        let input = Tensor::<NdArray, 3>::random(
            [16, 10, 4], // 16 sequences of 10 timesteps with 4 features each
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let output: Tensor<NdArray, 3> = mlp.forward(input);

        // Check output shape: [batch, sequence_length, output_dim]
        assert_eq!(output.shape().dims, [16, 10, 2]);
    }

    #[test]
    fn test_mlp_no_hidden_layers() {
        let device = NdArrayDevice::default();

        // Direct connection: 4 → 2
        let config = MLPConfig::new(4, vec![], 2);
        let mlp = config.init::<NdArray>(&device);

        let input = Tensor::<NdArray, 2>::random([1, 4], burn::tensor::Distribution::Default, &device);
        let output = mlp.forward(input);

        assert_eq!(output.shape().dims, [1, 2]);
    }

    #[test]
    fn test_mlp_forward_tanh() {
        let device = NdArrayDevice::default();

        let config = MLPConfig::new(4, vec![64], 2);
        let mlp = config.init::<NdArray>(&device);

        let input = Tensor::<NdArray, 2>::random([3, 4], burn::tensor::Distribution::Default, &device);
        let output: Tensor<NdArray, 2> = mlp.forward_tanh(input);

        // Check output shape
        assert_eq!(output.shape().dims, [3, 2]);

        // Check that output is in [-1, 1] (tanh bounds)
        let data = output.to_data();
        for &value in data.as_slice::<f32>().unwrap() {
            assert!(value >= -1.0 && value <= 1.0, "Tanh output should be in [-1, 1], got {}", value);
        }
    }
}
