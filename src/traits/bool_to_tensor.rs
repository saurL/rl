use burn::{
    prelude::*,
    tensor::{backend::Backend, TensorData}, // TensorData remplace Data
};

pub trait BoolToTensor<B: Backend> {
    fn to_bool_tensor(self, device: &B::Device) -> Tensor<B, 1, Bool>;
}

impl<B: Backend> BoolToTensor<B> for Vec<bool> {
    fn to_bool_tensor(self, device: &B::Device) -> Tensor<B, 1, Bool> {
        let len = self.len();
        let int_data: Vec<i32> = self.into_iter().map(|b| if b { 1 } else { 0 }).collect();
        let int_tensor: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(int_data, [len]), device);
        int_tensor.greater_elem(0)
    }
}
