mod model;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use model::{ActorConfig, CriticConfig};
use once_cell::sync::Lazy;
use rl::{
    algo::sac::{SACAgent, SACAgentConfig},
    gym::Pendulum,
};

type SACBackend = Autodiff<Wgpu>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

fn main() {

    // Create environment
    let mut env = Pendulum::new(200);

    // Network configuration
    let state_dim = 3; // [cos(θ), sin(θ), θ_dot]
    let action_dim = 1; // torque

    // Create networks
    let actor = ActorConfig::new(state_dim, action_dim)
        .with_hidden_size(64)
        .init::<SACBackend>(&*DEVICE);

    let critic1 = CriticConfig::new(state_dim, action_dim)
        .with_hidden_size(256)
        .init::<SACBackend>(&*DEVICE);

    let critic2 = CriticConfig::new(state_dim, action_dim)
        .with_hidden_size(256)
        .init::<SACBackend>(&*DEVICE);

    // Create SAC agent with configuration
    let config = SACAgentConfig {
        memory_capacity: 100_000,
        memory_batch_size: 256,
        use_prioritized_memory: false,
        gamma: 0.99,
        tau: 0.005,
        lr_actor: 3e-4,
        lr_critic: 3e-4,
        lr_alpha: 3e-4,
        auto_alpha: true,
        initial_alpha: 0.2,
        target_entropy: None, // Will default to -action_dim = -1.0
        target_update_interval: 1,
        actor_update_interval: 2,
        gradient_steps: 1,
        learning_starts: 1000,
        ..Default::default()
    };

    let mut agent = SACAgent::new(actor, critic1, critic2, &env, config, &*DEVICE);

    println!("Starting SAC training on Pendulum environment...");
    println!("Configuration:");
    println!("  - State dim: {}", state_dim);
    println!("  - Action dim: {}", action_dim);
    println!("  - Max episode length: 200");
    println!("  - Memory capacity: 100,000");
    println!("  - Batch size: 256");
    println!("  - Auto alpha tuning: enabled");
    println!();

    // Train the agent
    agent.go(&mut env);

    println!("Training complete!");
}
