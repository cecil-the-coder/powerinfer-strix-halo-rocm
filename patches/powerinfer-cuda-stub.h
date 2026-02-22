// Stub for PowerInfer CUDA/HIP stream integration
// Provides reset_powerinfer_default_cuda_stream() called in ggml_backend_cuda_context destructor.
// The SmallThinker MoE pipeline runs on CPU; no GPU stream state to reset.
static inline void reset_powerinfer_default_cuda_stream() {}
