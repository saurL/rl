use rand::{thread_rng, Rng};
use std::f32::consts::PI;

use crate::env::{ContinuousActionSpace, Environment, Report};

const MAX_SPEED: f32 = 8.0;
const MAX_TORQUE: f32 = 2.0;
const DT: f32 = 0.05;
const G: f32 = 10.0;
const M: f32 = 1.0;
const L: f32 = 1.0;

/// State representation: [cos(θ), sin(θ), θ_dot]
pub type PendulumState = [f32; 3];

/// Action: continuous torque in [-MAX_TORQUE, MAX_TORQUE]
pub type PendulumAction = [f32; 1];

/// Classic Pendulum environment with continuous action space
///
/// The goal is to keep the pendulum upright by applying torque.
/// The state is represented as [cos(θ), sin(θ), angular_velocity] to avoid
/// discontinuity issues with angle wrapping.
///
/// # Physics
/// - Mass: 1.0 kg
/// - Length: 1.0 m
/// - Gravity: 10.0 m/s²
/// - Time step: 0.05 s
/// - Max angular velocity: 8.0 rad/s
/// - Max torque: 2.0 N⋅m
///
/// # Reward
/// r = -θ² - 0.1⋅θ̇² - 0.001⋅u²
///
/// Where θ is the angle from vertical (0 = upright), θ̇ is angular velocity,
/// and u is the applied torque.
#[derive(Debug, Clone)]
pub struct Pendulum {
    theta: f32,
    theta_dot: f32,
    steps: usize,
    max_steps: usize,
    pub report: Report,
}

impl Pendulum {
    /// Create a new Pendulum environment
    ///
    /// # Arguments
    /// * `max_steps` - Maximum number of steps per episode (typically 200)
    pub fn new(max_steps: usize) -> Self {
        Self {
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_steps,
            report: Report::new(vec!["reward"]),
        }
    }

    fn get_state(&self) -> PendulumState {
        [self.theta.cos(), self.theta.sin(), self.theta_dot]
    }

    fn angle_normalize(x: f32) -> f32 {
        ((x + PI) % (2.0 * PI)) - PI
    }
}

impl Environment for Pendulum {
    type State = PendulumState;
    type Action = PendulumAction;

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        let torque = action[0].clamp(-MAX_TORQUE, MAX_TORQUE);

        // Physics: θ̈ = (3g/2L)sin(θ) + (3/mL²)u
        let theta_acc =
            (3.0 * G / (2.0 * L)) * self.theta.sin() + (3.0 / (M * L * L)) * torque;

        self.theta_dot += theta_acc * DT;
        self.theta_dot = self.theta_dot.clamp(-MAX_SPEED, MAX_SPEED);
        self.theta += self.theta_dot * DT;
        self.theta = Self::angle_normalize(self.theta);

        // Reward: -θ² - 0.1⋅θ̇² - 0.001⋅u²
        let reward = -(self.theta.powi(2)
            + 0.1 * self.theta_dot.powi(2)
            + 0.001 * torque.powi(2));

        self.steps += 1;
        self.report
            .entry("reward")
            .and_modify(|x| *x += reward as f64);

        let next_state = if self.steps >= self.max_steps {
            None
        } else {
            Some(self.get_state())
        };

        (next_state, reward)
    }

    fn reset(&mut self) -> Self::State {
        let mut rng = thread_rng();
        self.theta = rng.gen_range(-PI..PI);
        self.theta_dot = rng.gen_range(-1.0..1.0);
        self.steps = 0;
        self.get_state()
    }

    fn random_action(&self) -> Self::Action {
        let mut rng = thread_rng();
        [rng.gen_range(-MAX_TORQUE..MAX_TORQUE)]
    }

    fn is_active(&self) -> bool {
        self.steps < self.max_steps
    }
}

impl ContinuousActionSpace for Pendulum {
    fn action_dim(&self) -> usize {
        1
    }

    fn action_bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        Some((vec![-MAX_TORQUE], vec![MAX_TORQUE]))
    }
}

// Note: ToTensor implementations for Vec<[f32; N]> are already provided by
// the generic implementation in src/traits/to_tensor.rs, so we don't need
// to implement them here for PendulumState and PendulumAction.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pendulum_physics() {
        let mut env = Pendulum::new(200);

        // Test angle normalization
        let state = env.reset();
        assert!(state[0].abs() <= 1.0, "cos(θ) should be in [-1, 1]");
        assert!(state[1].abs() <= 1.0, "sin(θ) should be in [-1, 1]");
        assert!(
            state[2].abs() <= MAX_SPEED,
            "Angular velocity should be bounded"
        );

        // Test step with zero torque
        let (next_state, _) = env.step([0.0]);
        assert!(next_state.is_some(), "Should not be terminal after one step");

        // Test action clamping
        let (_, reward_high) = env.step([100.0]); // Should be clamped to MAX_TORQUE
        let (_, reward_low) = env.step([-100.0]); // Should be clamped to -MAX_TORQUE
        assert!(reward_high.is_finite());
        assert!(reward_low.is_finite());
    }

    #[test]
    fn pendulum_reward() {
        let mut env = Pendulum::new(200);
        env.theta = 0.0; // Upright
        env.theta_dot = 0.0; // Stationary

        let (_, reward) = env.step([0.0]); // No torque
        assert!(
            reward > -1.0,
            "Reward should be close to 0 when upright and stationary"
        );

        env.theta = PI; // Downward
        env.theta_dot = 0.0;
        let (_, reward_down) = env.step([0.0]);
        assert!(
            reward_down < reward,
            "Reward should be lower when pendulum is down"
        );
    }

    #[test]
    fn pendulum_action_bounds() {
        let env = Pendulum::new(200);
        assert_eq!(env.action_dim(), 1);

        let bounds = env.action_bounds().unwrap();
        assert_eq!(bounds.0, vec![-MAX_TORQUE]);
        assert_eq!(bounds.1, vec![MAX_TORQUE]);
    }

    #[test]
    fn pendulum_episode_length() {
        let mut env = Pendulum::new(5);
        env.reset();

        for i in 0..5 {
            let (next_state, _) = env.step([0.0]);
            if i < 4 {
                assert!(next_state.is_some(), "Should not be terminal yet");
            } else {
                assert!(next_state.is_none(), "Should be terminal at max_steps");
            }
        }
    }
}
