//! CUDA runtime for inference: context management, streams, memory allocation,
//! kernel loading, cuBLASLt GEMM, and NCCL communication.

pub mod context;
pub mod stream;
pub mod memory;
pub mod kernels;
pub mod gemm;
pub mod nccl;

// Re-export key cudarc types so consumers don't need `cudarc` directly.
pub use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, DeviceRepr,
};
pub use cudarc::driver::sys::CUfunction_attribute_enum;
pub use cudarc::driver::safe::CudaView;
pub use cudarc::driver::safe::PushKernelArg;
pub use cudarc::cublaslt::safe::{
    Activation, CudaBlasLT, MatmulConfig as CublasMatmulConfig,
};
pub use cudarc::nccl::safe::{
    Comm as NcclComm, ReduceOp as NcclReduceOp, NcclType,
};
pub use cudarc::nccl::result::{group_end, group_start};
