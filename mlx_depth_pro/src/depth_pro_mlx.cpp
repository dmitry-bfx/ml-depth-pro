#include "depth_pro_mlx.h"
#include "model.h"
#include <cstring>
#include <filesystem>
#include "mlx/memory.h"

struct DepthProMLX {
    depth_pro::DepthProModel model;
};

extern "C" {

DepthProMLX* depth_pro_mlx_create(const char* weights_path) {
    try {
        // MLX aborts (rather than returning error) if file doesn't exist
        if (!std::filesystem::exists(weights_path)) {
            fprintf(stderr, "depth_pro_mlx_create error: file not found: %s\n", weights_path);
            return nullptr;
        }
        auto* ctx = new DepthProMLX();
        ctx->model.load(weights_path);
        return ctx;
    } catch (const std::exception& e) {
        fprintf(stderr, "depth_pro_mlx_create error: %s\n", e.what());
        return nullptr;
    }
}

void depth_pro_mlx_destroy(DepthProMLX* model) {
    delete model;
}

int depth_pro_mlx_forward(
    DepthProMLX* model,
    const float* patch_enc_out,
    const float* hook0_out,
    const float* hook1_out,
    const float* image_enc_out,
    const float* fov_enc_out,
    float* depth_out,
    float* fov_deg_out) {

    namespace mx = mlx::core;
    try {
        // Wrap input pointers as MLX arrays (iterator constructor: array(It, Shape, Dtype))
        auto make_array = [](const float* data, mx::Shape shape) -> mx::array {
            return mx::array(data, shape, mx::float32);
        };

        auto patch_enc = make_array(patch_enc_out, {35, 577, 1024});
        auto hook0 = make_array(hook0_out, {35, 577, 1024});
        auto hook1 = make_array(hook1_out, {35, 577, 1024});
        auto image_enc = make_array(image_enc_out, {1, 577, 1024});
        auto fov_enc = make_array(fov_enc_out, {1, 577, 1024});

        auto [depth, fov_deg] = model->model.forward(
            patch_enc, hook0, hook1, image_enc, fov_enc);

        // Force evaluation
        mx::eval(depth);
        mx::eval(fov_deg);

        // depth is [1, 1536, 1536, 1] in NHWC -> copy to output as [1, 1, 1536, 1536] NCHW
        // transpose creates a non-contiguous view — must flatten/contiguous before reading
        auto depth_nchw = mx::flatten(mx::transpose(depth, {0, 3, 1, 2}));
        mx::eval(depth_nchw);

        auto depth_data = depth_nchw.data<float>();
        std::memcpy(depth_out, depth_data, 1 * 1 * 1536 * 1536 * sizeof(float));

        // FOV deg is [1, 1, 1, 1] -> scalar
        auto fov_data = fov_deg.data<float>();
        *fov_deg_out = fov_data[0];

        // Clear Metal buffer cache to prevent memory accumulation across calls
        mx::clear_cache();

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "depth_pro_mlx_forward error: %s\n", e.what());
        return -1;
    }
}

} // extern "C"
