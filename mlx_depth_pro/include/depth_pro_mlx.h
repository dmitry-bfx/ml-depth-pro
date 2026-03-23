#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DepthProMLX DepthProMLX;

// Create model and load weights from safetensors file.
DepthProMLX* depth_pro_mlx_create(const char* weights_path);

// Free model resources.
void depth_pro_mlx_destroy(DepthProMLX* model);

// Run the post-ViT portion of Depth Pro.
// All inputs are float32.
// Returns 0 on success, non-zero on error.
int depth_pro_mlx_forward(
    DepthProMLX* model,
    const float* patch_enc_out,   // [35, 577, 1024]
    const float* hook0_out,       // [35, 577, 1024]
    const float* hook1_out,       // [35, 577, 1024]
    const float* image_enc_out,   // [1, 577, 1024]
    const float* fov_enc_out,     // [1, 577, 1024]
    float* depth_out,             // [1, 1, 1536, 1536] caller-allocated
    float* fov_deg_out            // scalar output
);

#ifdef __cplusplus
}
#endif
