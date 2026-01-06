// NEW API DEMO: Using library's generic MLP instead of custom models
// Users no longer need to implement TD3ActorModel and TD3CriticModel traits
// Just configure the network architecture and the library handles the rest!

use burn::backend::{Autodiff, NdArray};
use once_cell::sync::Lazy;
#[cfg(feature = "viz")]
use rl::viz;
use rl::{
    algo::td3::{MLPCritic, TD3Agent, TD3AgentConfig},
    env::Environment,
    gym::Pendulum,
    nn::MLPConfig,
    traits::TrainableAgent,
};

type TD3Backend = Autodiff<NdArray>;

static DEVICE: Lazy<<NdArray as burn::prelude::Backend>::Device> = Lazy::new(Default::default);

const NUM_EPISODES: u16 = 200;
const MAX_STEPS_PER_EPISODE: usize = 200;

fn main() {
    // Create environment
    let mut env = Pendulum::new(200);

    // Network configuration
    let state_dim = 3; // [cos(θ), sin(θ), θ_dot]
    let action_dim = 1; // torque

    // NEW API: Just configure the architecture - no custom model implementation needed!
    // Actor: state (3) → hidden (256, 256) → action (1) with tanh
    // Hidden layers use ReLU, output layer uses tanh (bounded actions [-1, 1])
    let actor = MLPConfig::new(state_dim, vec![256, 256], action_dim)
        .init::<TD3Backend>(&*DEVICE);

    // Critic 1 & 2: concatenate [state (3), action (1)] → hidden (256, 256) → Q-value (1)
    // Hidden layers use ReLU, output layer is linear
    let critic1 = MLPCritic::new(state_dim, action_dim, vec![256, 256], &*DEVICE);
    let critic2 = MLPCritic::new(state_dim, action_dim, vec![256, 256], &*DEVICE);

    // Create TD3 agent with configuration
    let config = TD3AgentConfig {
        memory_capacity: 50_000,
        memory_batch_size: 128,
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
    };

    let mut agent: TD3Agent<TD3Backend, _, _, Pendulum, 2> =
        TD3Agent::new(actor, critic1, critic2, &env, config, &*DEVICE);

    #[cfg(feature = "viz")]
    let viz_keys = vec!["reward", "steps"];
    #[cfg(feature = "viz")]
    let (handle, tx) = viz::init(&viz_keys, NUM_EPISODES);

    let mut window_rewards = Vec::new();

    // Train the agent
    for episode in 0..NUM_EPISODES {
        env.reset();
        let mut episode_reward = 0.0;
        let mut steps = 0;

        loop {
            let (reward, done, _metrics) = agent.step(&mut env);
            episode_reward += reward;
            steps += 1;

            if done || steps >= MAX_STEPS_PER_EPISODE {
                break;
            }
        }

        // Rolling average of last 10 episodes
        window_rewards.push(episode_reward);
        if window_rewards.len() > 10 {
            window_rewards.remove(0);
        }

        let avg_reward = window_rewards.iter().sum::<f32>() / window_rewards.len() as f32;

        // Print progress every 10 episodes
        if (episode + 1) % 10 == 0 {
            println!(
                "Episode {} | Reward: {:.2} | Steps: {}",
                episode + 1, avg_reward, steps
            );
        }

        // Send to viz
        #[cfg(feature = "viz")]
        {
            if let Err(e) = tx.send(viz::Update {
                episode: episode as u16,
                data: vec![avg_reward as f64, steps as f64],
            }) {
                eprintln!("Failed to send to viz: {:?}", e);
            }
        }
    }

    #[cfg(feature = "viz")]
    let _ = handle.join();

    println!("\nTraining complete!");
}
