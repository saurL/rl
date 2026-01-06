// NEW API DEMO: Using library's generic MLP instead of custom models
// Users no longer need to implement PPOActorModel and PPOCriticModel traits
// Just configure the network architecture and the library handles the rest!

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use gym_rs::utils::renderer::RenderMode;
use once_cell::sync::Lazy;
#[cfg(feature = "viz")]
use rl::viz;
use rl::{
    algo::ppo::{PPOAgent, PPOAgentConfig},
    env::Environment,
    gym::CartPole,
    nn::MLPConfig,
    traits::TrainableAgent,
};



type PPOBackend = Autodiff<Wgpu>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const NUM_EPISODES: u16 = 500;

fn main() {
   

    // Create environment
    let mut env = CartPole::new(RenderMode::Human);

    // Network configuration
    let state_dim = 4; // [x, x_dot, theta, theta_dot]
    let num_actions = 2; // left or right

    // NEW API: Just configure the architecture - no custom model implementation needed!
    // Actor: state (4) → hidden (64) → hidden (64) → actions (2)
    // Hidden layers use ReLU, output layer is linear (no activation)
    let actor = MLPConfig::new(state_dim, vec![64, 64], num_actions)
        .init::<PPOBackend>(&*DEVICE);

    // Critic: state (4) → hidden (64) → hidden (64) → value (1)
    // Hidden layers use ReLU, output layer is linear (no activation)
    let critic = MLPConfig::new(state_dim, vec![64, 64], 1)
        .init::<PPOBackend>(&*DEVICE);

    // Create PPO agent with configuration
    let config = PPOAgentConfig {
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
    };

    let mut agent: PPOAgent<PPOBackend, _, _, CartPole, 2> =
        PPOAgent::new(actor, critic, config, &*DEVICE);

    // Initialize visualization with extended metrics
    #[cfg(feature = "viz")]
    let viz_keys = vec![
        "reward",
        "steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
    ];
    #[cfg(feature = "viz")]
    let (handle, tx) = viz::init(&viz_keys, NUM_EPISODES);

    // Metrics tracking for averaging every 10 episodes
    let mut window_rewards = Vec::new();
    let mut window_policy_losses = Vec::new();
    let mut window_value_losses = Vec::new();
    let mut window_entropies = Vec::new();
    let mut window_approx_kls = Vec::new();
    let mut window_clip_fractions = Vec::new();
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

            if done {
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
            let avg_approx_kl = episode_metrics
                .iter()
                .map(|m| m.approx_kl.unwrap_or(0.0))
                .sum::<f32>()
                / n;
            let avg_clip_frac = episode_metrics
                .iter()
                .map(|m| m.clip_fraction.unwrap_or(0.0))
                .sum::<f32>()
                / n;

            window_policy_losses.push(avg_policy_loss);
            window_value_losses.push(avg_value_loss);
            window_entropies.push(avg_entropy);
            window_approx_kls.push(avg_approx_kl);
            window_clip_fractions.push(avg_clip_frac);

            if window_policy_losses.len() > 10 {
                window_policy_losses.remove(0);
                window_value_losses.remove(0);
                window_entropies.remove(0);
                window_approx_kls.remove(0);
                window_clip_fractions.remove(0);
            }
        }

        // Compute rolling averages (mean of last N episodes, where N <= 10)
        let avg_reward = window_rewards.iter().sum::<f32>() / window_rewards.len() as f32;
        let avg_steps = window_steps.iter().sum::<usize>() as f32 / window_steps.len() as f32;

        let (avg_policy_loss, avg_value_loss, avg_entropy, avg_kl, avg_clip) =
            if !window_policy_losses.is_empty() {
                (
                    window_policy_losses.iter().sum::<f32>() / window_policy_losses.len() as f32,
                    window_value_losses.iter().sum::<f32>() / window_value_losses.len() as f32,
                    window_entropies.iter().sum::<f32>() / window_entropies.len() as f32,
                    window_approx_kls.iter().sum::<f32>() / window_approx_kls.len() as f32,
                    window_clip_fractions.iter().sum::<f32>()
                        / window_clip_fractions.len() as f32,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };



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
                avg_kl as f64,
                avg_clip as f64,
            ];

            let _ = tx.send(viz::Update {
                episode: episode as u16,
                data,
            });
        }
    }

    // Wait for visualization thread to finish
    #[cfg(feature = "viz")]
    let _ = handle.join();

    println!("Training complete!");
}
