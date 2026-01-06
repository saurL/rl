// NEW API DEMO: Using library's generic MLP instead of custom models
// Users no longer need to implement DQNModel trait
// Just configure the network architecture and the library handles the rest!

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use gym_rs::utils::renderer::RenderMode;
use once_cell::sync::Lazy;
use rl::{
    algo::dqn::{DQNAgent, DQNAgentConfig},
    gym::CartPole,
    nn::MLPConfig,
    viz,
};

type DQNBackend = Autodiff<Wgpu>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const NUM_EPISODES: u16 = 500;

fn main() {
    let mut env = CartPole::new(RenderMode::Human);

    // NEW API: Just configure the architecture - no custom model implementation needed!
    // Network: state (4) → hidden (64) → hidden (128) → Q-values (2)
    // Hidden layers use ReLU, output layer is linear (no activation)
    let model = MLPConfig::new(4, vec![64, 128], 2).init::<DQNBackend>(&*DEVICE);

    let agent_config = DQNAgentConfig::default();
    let mut agent = DQNAgent::new(model, agent_config, &*DEVICE);

    let (handle, tx) = viz::init(env.report.keys(), NUM_EPISODES);

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        let report = env.report.take();

        // Print progress every 50 episodes
        if (i + 1) % 50 == 0 {
            let reward = report.get("reward").copied().unwrap_or(0.0);
            println!("Episode {}/{} | Reward: {:.2}", i + 1, NUM_EPISODES, reward);
        }

        tx.send(viz::Update {
            episode: i,
            data: report.values().copied().collect(),
        })
        .unwrap();
    }

    let _ = handle.join();
    println!("\nTraining complete!");
}
