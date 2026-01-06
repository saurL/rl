use burn::{
    prelude::*,
    tensor::{backend::Backend, BasicOps, Element, TensorData}, // TensorData remplace Data
};

/// A trait for converting items to tensors
///
/// Commonly implemented for `Vec<T>` to convert batches of `T` to a tensor of dimension `D`
///
/// See implementations of this for [`CartPole`](crate::gym::CartPole) as an example of how to implement this trait
pub trait ToTensor<B: Backend, const D: usize, K: BasicOps<B>> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, D, K>;
}

// Marker trait to restrict blanket implementations
trait IntoData<const D: usize> {}

impl<const D: usize> IntoData<D> for TensorData {}
impl<const D: usize> IntoData<D> for &TensorData {}

impl<E, const A: usize, const B: usize, const C: usize, const D: usize> IntoData<4>
    for [[[[E; D]; C]; B]; A]
{
}
impl<E, const A: usize, const B: usize, const C: usize> IntoData<3> for [[[E; C]; B]; A] {}
impl<E, const A: usize, const B: usize> IntoData<2> for [[E; B]; A] {}
impl<E, const A: usize> IntoData<1> for [E; A] {}

impl<B, const D: usize, K, T> ToTensor<B, D, K> for T
where
    B: Backend,
    K: BasicOps<B>,
    T: Into<TensorData> + IntoData<D>, // Utilise TensorData au lieu de Data
{
    fn to_tensor(self, device: &<B as Backend>::Device) -> Tensor<B, D, K> {
        Tensor::from_data(self.into(), device)
    }
}

// Implementations

impl<B, E, K> ToTensor<B, 1, K> for Vec<E>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    #[inline]
    fn to_tensor(self, device: &<B as Backend>::Device) -> Tensor<B, 1, K> {
        let len = self.len();
        Tensor::from_data(TensorData::new(self, [len]), device)
    }
}

impl<B, E, K, const A: usize> ToTensor<B, 2, K> for Vec<[E; A]>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    #[inline]
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, K> {
        let batch_size = self.len();
        // Pre-allocate exact capacity to avoid reallocation
        let mut flat = Vec::with_capacity(batch_size * A);

        // Use extend_from_slice instead of flatten().collect() for better performance
        for array in self.iter() {
            flat.extend_from_slice(array);
        }

        let data = TensorData::new(flat, [batch_size, A]);
        Tensor::<B, 2, K>::from_data(data, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_vec_f32_to_tensor_1d() {
        let device = NdArrayDevice::default();
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tensor: Tensor<NdArray, 1> = data.to_tensor(&device);

        assert_eq!(tensor.shape().dims, [4]);
        let tensor_data = tensor.to_data();
        assert_eq!(tensor_data.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec_array_to_tensor_2d() {
        let device = NdArrayDevice::default();

        // Simule batch.states de CartPole (Vec<[f32; 4]>)
        let states = vec![
            [1.0_f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];

        let tensor: Tensor<NdArray, 2> = states.to_tensor(&device);

        // Doit avoir la forme [3, 4] (3 états, 4 dimensions chacun)
        assert_eq!(tensor.shape().dims, [3, 4]);

        let tensor_data = tensor.to_data();
        let expected = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        assert_eq!(tensor_data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_vec_array_to_tensor_2d_different_size() {
        let device = NdArrayDevice::default();

        // Test avec une taille différente
        // Note: NdArray backend utilise i64 pour Int, donc on teste avec i64
        let data = vec![
            [10_i64, 20],
            [30, 40],
        ];

        let tensor: Tensor<NdArray, 2, Int> = data.to_tensor(&device);

        assert_eq!(tensor.shape().dims, [2, 2]);

        let tensor_data = tensor.to_data();
        assert_eq!(tensor_data.as_slice::<i64>().unwrap(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_single_element_vec_to_tensor_2d() {
        let device = NdArrayDevice::default();

        let states = vec![[1.0_f32, 2.0, 3.0]];
        let tensor: Tensor<NdArray, 2> = states.to_tensor(&device);

        assert_eq!(tensor.shape().dims, [1, 3]);
    }
}
