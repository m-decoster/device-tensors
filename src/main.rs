#![feature(const_generics)]

use tch;

pub trait Device {
    fn tch_device() -> tch::Device;
}

/// Device indicator for variables on the CPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuDevice;
impl Device for CpuDevice {
    fn tch_device() -> tch::Device {
        tch::Device::Cpu
    }
}
/// Device indicator for variables on a GPU (CUDA required).
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaDevice<const N: usize>;
impl<const N: usize> Device for CudaDevice<N> {
    fn tch_device() -> tch::Device {
        tch::Device::Cuda(N)
    }
}

#[derive(Debug)]
pub struct DeviceTensor<D> where D: Device {
    internal: tch::Tensor,
    _device_marker: std::marker::PhantomData<D>,
}

impl<D: Device> DeviceTensor<D> {
    pub fn randn(size: &[i64]) -> DeviceTensor<D> {
        DeviceTensor {
            internal: tch::Tensor::randn(size, (tch::Kind::Float, D::tch_device())),
            _device_marker: Default::default(),
        }
    }

    pub fn to_device<D2: Device>(self) -> DeviceTensor<D2> {
        DeviceTensor {
            internal: self.internal.to_device(D2::tch_device()),
            _device_marker: Default::default(),
        }
    }
}

impl<'a, 'b, D: Device> std::ops::Add<&'b DeviceTensor<D>> for &'a DeviceTensor<D> {
    type Output = DeviceTensor<D>;

    fn add(self, other: &'b DeviceTensor<D>) -> DeviceTensor<D> {
        DeviceTensor {
            internal: &self.internal + &other.internal,
            _device_marker: Default::default(),
        }
    }
}

fn main() {
    let tensor_1: DeviceTensor<CpuDevice> = DeviceTensor::randn(&[2, 3]);
    let tensor_2: DeviceTensor<CudaDevice<0>> = DeviceTensor::randn(&[2, 3]);
    let tensor_2_cpu = tensor_2.to_device::<CpuDevice>();

    // let result = &tensor_1 + &tensor_2; // Doesn't compile.
    let result = &tensor_1 + &tensor_2_cpu; // Compiles and runs without error!
}