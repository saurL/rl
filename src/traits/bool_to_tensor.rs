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
        // Convertir directement en Bool sans passer par Int et greater_elem
        let bool_data: Vec<u8> = self.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect();
        Tensor::from_data(TensorData::new(bool_data, [len]).convert::<B::BoolElem>(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_vec_bool_to_tensor() {
        let device = NdArrayDevice::default();
        let bools = vec![true, false, true, true, false];
        let tensor: Tensor<NdArray, 1, Bool> = bools.to_bool_tensor(&device);

        assert_eq!(tensor.shape().dims, [5]);

        // Convertir en int pour vérifier les valeurs
        // Note: NdArray backend utilise i64 pour Int
        let int_tensor = tensor.int();
        let data = int_tensor.to_data();
        assert_eq!(data.as_slice::<i64>().unwrap(), &[1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_vec_bool_all_true() {
        let device = NdArrayDevice::default();
        let bools = vec![true, true, true];
        let tensor: Tensor<NdArray, 1, Bool> = bools.to_bool_tensor(&device);

        assert_eq!(tensor.shape().dims, [3]);

        let int_tensor = tensor.int();
        let data = int_tensor.to_data();
        assert_eq!(data.as_slice::<i64>().unwrap(), &[1, 1, 1]);
    }

    #[test]
    fn test_vec_bool_all_false() {
        let device = NdArrayDevice::default();
        let bools = vec![false, false, false];
        let tensor: Tensor<NdArray, 1, Bool> = bools.to_bool_tensor(&device);

        assert_eq!(tensor.shape().dims, [3]);

        let int_tensor = tensor.int();
        let data = int_tensor.to_data();
        assert_eq!(data.as_slice::<i64>().unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn test_vec_bool_empty() {
        let device = NdArrayDevice::default();
        let bools: Vec<bool> = vec![];
        let tensor: Tensor<NdArray, 1, Bool> = bools.to_bool_tensor(&device);

        assert_eq!(tensor.shape().dims, [0]);
    }
}
