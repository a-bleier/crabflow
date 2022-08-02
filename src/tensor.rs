#[derive(Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<u32>,
    offsets: Vec<u32>,
}
#[derive(Debug, PartialEq)]
pub enum TensorError {
    WrongShapeError,
    OutOfBoundsError,
}

impl Tensor {
    pub fn zeros(dim: &[u32]) -> Result<Tensor, TensorError> {
        let mut data_size: u32 = 1;

        for val in dim {
            data_size *= val;
        }

        Tensor::new(
            &std::iter::repeat(0.0)
                .take(data_size as usize)
                .collect::<Vec<_>>(),
            dim,
        )
    }

    pub fn new(data: &[f32], shape: &[u32]) -> Result<Tensor, TensorError> {
        let dim_prod: u32 = shape.iter().product();
        println!("Data: {:?}", data);
        println!("Shape: {:?}", shape);
        if dim_prod as usize != data.len() {
            return Err(TensorError::WrongShapeError);
        }

        let mut offsets = Vec::new();

        let mut offset = 1;
        for d in shape[1..].iter().rev() {
            offset *= d;
            offsets.push(offset);
        }
        offsets.reverse();
        offsets.push(1);

        Ok(Tensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
            offsets: offsets.clone(),
        })
    }

    pub fn at(&self, idx_arr: &[u32]) -> Result<Tensor, TensorError> {
        if idx_arr.len() > self.shape.len() {
            return Err(TensorError::WrongShapeError);
        }

        let mut start_index: usize = 0;

        for (i, index_val) in idx_arr.iter().enumerate() {
            start_index += (index_val * self.offsets[i]) as usize;
        }

        let end_index: usize = start_index + self.offsets[idx_arr.len() - 1] as usize;
        let mut new_shape = self.shape[idx_arr.len()..].to_vec();
        if idx_arr.len() == self.shape.len() {
            new_shape = vec![1];
        }

        return Tensor::new(&self.data[start_index..end_index], &new_shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn new_tensor() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let tensor = Tensor::new(&data, &shape).unwrap();
        for (i, val) in tensor.data.iter().enumerate() {
            assert_eq!(true, *val == data[i], "not equal at data element {}", i);
        }
        for (i, val) in tensor.shape.iter().enumerate() {
            assert_eq!(true, *val == shape[i], "not equal at shape element {}", i);
        }

        assert_eq!(3, tensor.offsets.len());
        assert_eq!(6, tensor.offsets[0]);
        assert_eq!(3, tensor.offsets[1]);
    }

    #[test]
    fn new_bad_tensor() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![4, 2, 3];
        assert_eq!(
            TensorError::WrongShapeError,
            Tensor::new(&data, &shape).unwrap_err()
        );
    }

    #[test]
    fn zero_tensor() {
        let dim = vec![2, 2];
        let zeros = Tensor::zeros(&dim);
    }
    #[test]
    fn tensor_indexing_1d() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let tensor = Tensor::new(&data, &shape).unwrap();
        let new_tensor = tensor.at(&[1, 1, 0]).unwrap();
        assert_eq!(1, new_tensor.data.len());
        assert_eq!(9.0, new_tensor.data[0]);
    }

    #[test]
    fn tensor_indexing_md() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let tensor = Tensor::new(&data, &shape).unwrap();
        let new_tensor = tensor.at(&[1]).unwrap();

        let expected_shape: [u32; 2] = [2, 3];
        let expected_data: Vec<f32> = (6..12).map(|x| x as f32).collect();
        assert_eq!(
            0,
            expected_data
                .iter()
                .zip(new_tensor.data.iter())
                .filter(|&(a, b)| a != b)
                .count()
        );
        assert_eq!(
            0,
            expected_shape
                .iter()
                .zip(new_tensor.shape.iter())
                .filter(|&(a, b)| a != b)
                .count()
        );
    }
    #[test]
    fn tensor_indexing_outoufbounds() {
        //TODO Write a test which checks out of bounds indices
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let tensor = Tensor::new(&data, &shape).unwrap();
        let new_tensor = tensor.at(&[1]).unwrap();

        let expected_shape: [u32; 2] = [2, 3];
        let expected_data: Vec<f32> = (6..12).map(|x| x as f32).collect();
        assert_eq!(
            0,
            expected_data
                .iter()
                .zip(new_tensor.data.iter())
                .filter(|&(a, b)| a != b)
                .count()
        );
        assert_eq!(
            0,
            expected_shape
                .iter()
                .zip(new_tensor.shape.iter())
                .filter(|&(a, b)| a != b)
                .count()
        );
    }
    #[test]
    fn tensor_indexing_wrong_shape() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let tensor = Tensor::new(&data, &shape).unwrap();
        let tensor_err = tensor.at(&[2, 2, 2, 2]).unwrap_err();
        assert_eq!(TensorError::WrongShapeError, tensor_err);
    }
}
