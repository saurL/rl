// NEW API DEMO: Using library's generic MLP instead of custom models
// Users no longer need to implement A2CActorModel and A2CCriticModel traits
// Just configure the network architecture and the library handles the rest!

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use gym_rs::utils::renderer::RenderMode;
use once_cell::sync::Lazy;
#[cfg(feature = "viz")]
use rl::viz;
use rl::{
    algo::a2c::{A2CAgent, A2CAgentConfig},
    env::Environment,
    gym::CartPole,
    nn::MLPConfig,
    traits::TrainableAgent,
};

type A2CBackend = Autodiff<Wgpu>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const NUM_EPISODES: u16 = 1000;
const MAX_STEPS_PER_EPISODE: usize = 500;

fn main() {
    // Create environment
    let mut env = CartPole::new(RenderMode::Human);

    // Network configuration
    let state_dim = 4; // [x, x_dot, theta, theta_dot]
    let num_actions = 2; // left or right

    // NEW API: Just configure the architecture - no custom model implementation needed!
    // Actor: state (4) → hidden (128) → hidden (128) → actions (2)
    // Hidden layers use ReLU, output layer is linear (no activation)
    let actor = MLPConfig::new(state_dim, vec![128, 128], num_actions)
        .init::<A2CBackend>(&*DEVICE);

    // Critic: state (4) → hidden (128) → hidden (128) → value (1)
    // Hidden layers use ReLU, output layer is linear (no activation)
    let critic = MLPConfig::new(state_dim, vec![128, 128], 1)
        .init::<A2CBackend>(&*DEVICE);

    // Create A2C agent with configuration
    let config = A2CAgentConfig {
        gamma: 0.99,
        lr_actor: 7e-4,
        lr_critic: 7e-4,
        entropy_coef: 0.01,
        value_coef: 0.5,
        n_steps: 5,
        gradient_clip: Some(0.5),
    };

    let mut agent: A2CAgent<A2CBackend, _, _, CartPole, 2> =
        A2CAgent::new(actor, critic, config, &*DEVICE);

    // Initialize visualization with extended metrics
    #[cfg(feature = "viz")]
    let viz_keys = vec![
        "reward",
        "steps",
        "policy_loss",
        "value_loss",
        "entropy",
    ];
    #[cfg(feature = "viz")]
    let (handle, tx) = viz::init(&viz_keys, NUM_EPISODES);

    // Metrics tracking for averaging every 10 episodes
    let mut window_rewards = Vec::new();
    let mut window_policy_losses = Vec::new();
    let mut window_value_losses = Vec::new();
    let mut window_entropies = Vec::new();
    let mut window_steps = Vec::new();

    // Train the agent with TrainableAgent API (like PyTorch)
    for episode in 0..NUM_EPISODES {
        env.reset();

        let mut episode_reward = 0.0;
        let mut steps = 0;
        let mut episode_metrics = Vec::new();

        // Training loop (like PyTorch training loop)
        loop {
            // Forward pass: collect experience
            let (reward, done, metrics) = agent.step(&mut env);

            episode_reward += reward;
            steps += 1;

            if let Some(metrics) = metrics {
                episode_metrics.push(metrics);
            }

            // Break if episode is done or max steps reached
            if done || steps >= MAX_STEPS_PER_EPISODE {
                break;
            }
        }

        // Track episode metrics (keep last 10 episodes as rolling window)
        window_rewards.push(episode_reward);
        window_steps.push(steps);
        if window_rewards.len() > 10 {
            window_rewards.remove(0);
            window_steps.remove(0);
        }

        // Aggregate training metrics for this episode
        if !episode_metrics.is_empty() {
            let n = episode_metrics.len() as f32;
            let avg_policy_loss = episode_metrics.iter().map(|m| m.policy_loss).sum::<f32>() / n;
            let avg_value_loss = episode_metrics.iter().map(|m| m.value_loss).sum::<f32>() / n;
            let avg_entropy = episode_metrics.iter().map(|m| m.entropy).sum::<f32>() / n;

            window_policy_losses.push(avg_policy_loss);
            window_value_losses.push(avg_value_loss);
            window_entropies.push(avg_entropy);

            if window_policy_losses.len() > 10 {
                window_policy_losses.remove(0);
                window_value_losses.remove(0);
                window_entropies.remove(0);
            }
        }

        // Compute rolling averages (mean of last N episodes, where N <= 10)
        let avg_reward = window_rewards.iter().sum::<f32>() / window_rewards.len() as f32;
        let avg_steps = window_steps.iter().sum::<usize>() as f32 / window_steps.len() as f32;

        let (avg_policy_loss, avg_value_loss, avg_entropy) = if !window_policy_losses.is_empty() {
            (
                window_policy_losses.iter().sum::<f32>() / window_policy_losses.len() as f32,
                window_value_losses.iter().sum::<f32>() / window_value_losses.len() as f32,
                window_entropies.iter().sum::<f32>() / window_entropies.len() as f32,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // Print progress every 10 episodes
        if (episode + 1) % 10 == 0 {
            println!(
                "Episode {} | Reward: {:.2} | Steps: {:.1} | Policy Loss: {:.6} | Value Loss: {:.6} | Entropy: {:.4}",
                episode + 1, avg_reward, avg_steps, avg_policy_loss, avg_value_loss, avg_entropy
            );
        }

        // Send rolling average to viz at every episode
        #[cfg(feature = "viz")]
        {
            // Build data vector in the same order as viz_keys
            let data = vec![
                avg_reward as f64,
                avg_steps as f64,
                avg_policy_loss as f64,
                avg_value_loss as f64,
                avg_entropy as f64,
            ];

            if let Err(e) = tx.send(viz::Update {
                episode: episode as u16,
                data,
            }) {
                eprintln!("Failed to send to viz: {:?}", e);
            }
        }
    }

    // Wait for visualization thread to finish
    #[cfg(feature = "viz")]
    let _ = handle.join();

    println!("\nTraining complete!");
}
