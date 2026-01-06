use burn::prelude::*;
use gym_rs::core::{ActionReward, Env};
use gym_rs::envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation};
use gym_rs::utils::renderer::RenderMode;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::{DiscreteActionSpace, Environment, Report};
use crate::traits::ToTensor;

fn obs2arr(observation: MountainCarObservation) -> [f32; 2] {
    Vec::from(observation)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("vec is length 2")
}

/// Actions for the [`MountainCar`] environment
/// 0 = push left, 1 = no push, 2 = push right
#[derive(FromRepr, EnumIter, VariantArray, Clone, Copy, Debug)]
pub enum MCAction {
    PushLeft = 0,
    NoPush = 1,
    PushRight = 2,
}

impl From<usize> for MCAction {
    fn from(value: usize) -> Self {
        Self::from_repr(value).expect("MCAction::from is only called with valid values [0, 1, 2]")
    }
}

impl<B: Backend> ToTensor<B, 2, Int> for Vec<MCAction> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Int> {
        let len = self.len();
        let data = TensorData::new(
            self.into_iter().map(|x| x as i32).collect::<Vec<_>>(),
            [len, 1],
        );
        Tensor::<B, 2, Int>::from_data(data.convert::<B::IntElem>(), device)
    }
}

/// The classic Mountain Car reinforcement learning environment with discrete actions
///
/// This implementation is a thin wrapper around [gym_rs](https://github.com/MathisWellmann/gym-rs)
#[derive(Debug, Clone)]
pub struct MountainCar {
    gym_env: MountainCarEnv,
    pub report: Report,
}

impl MountainCar {
    pub fn new(render_mode: RenderMode) -> Self {
        Self {
            gym_env: MountainCarEnv::new(render_mode),
            report: Report::new(vec!["reward"]),
        }
    }
}

impl Environment for MountainCar {
    type State = [f32; 2];  // [position, velocity]
    type Action = MCAction;

    fn random_action(&self) -> Self::Action {
        MCAction::iter().choose(&mut thread_rng()).unwrap()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        let ActionReward {
            observation,
            reward,
            done,
            ..
        } = self.gym_env.step(action as usize);

        let next_state = if done {
            None
        } else {
            Some(obs2arr(observation))
        };

        self.report.entry("reward").and_modify(|x| *x += *reward);

        (next_state, *reward as f32)
    }

    fn reset(&mut self) -> Self::State {
        obs2arr(self.gym_env.reset(None, false, None).0)
    }

    fn current_state(&self) -> Self::State {
        obs2arr(self.gym_env.state)
    }
}

impl DiscreteActionSpace for MountainCar {
    fn actions(&self) -> Vec<Self::Action> {
        MCAction::VARIANTS.to_vec()
    }
}
