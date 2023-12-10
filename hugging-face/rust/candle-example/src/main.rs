use candle_core::backend::BackendDevice;
use candle_core::cpu_backend::CpuDevice;
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Is is currently not possible to set the seed for the CPU backend.
    //CpuDevice.set_seed(42).expect("Failed to set seed");

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
