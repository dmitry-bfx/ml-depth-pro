#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace depth_pro {

namespace mx = mlx::core;

// Convenience: load a weight by key, assert it exists.
inline mx::array get_weight(
    const std::unordered_map<std::string, mx::array>& w,
    const std::string& key) {
    auto it = w.find(key);
    if (it == w.end()) {
        throw std::runtime_error("Missing weight: " + key);
    }
    return it->second;
}

// Conv2d forward. MLX expects input [N,H,W,C], weight [O,kH,kW,I].
// Weight is already stored in OHWI layout from conversion script.
mx::array conv2d_forward(
    const mx::array& input,
    const mx::array& weight,
    std::pair<int,int> stride = {1,1},
    std::pair<int,int> padding = {0,0});

// Conv2d forward with bias.
mx::array conv2d_forward(
    const mx::array& input,
    const mx::array& weight,
    const mx::array& bias,
    std::pair<int,int> stride = {1,1},
    std::pair<int,int> padding = {0,0});

// ConvTranspose2d forward. Weight in OHWI layout.
mx::array conv_transpose2d_forward(
    const mx::array& input,
    const mx::array& weight,
    std::pair<int,int> stride = {2,2},
    std::pair<int,int> padding = {0,0});

// ConvTranspose2d forward with bias.
mx::array conv_transpose2d_forward(
    const mx::array& input,
    const mx::array& weight,
    const mx::array& bias,
    std::pair<int,int> stride = {2,2},
    std::pair<int,int> padding = {0,0});

// ReLU
mx::array relu(const mx::array& x);

// Remove CLS token and reshape ViT output from [N,577,1024] to [N,24,24,1024] (NHWC).
mx::array reshape_feature(const mx::array& embeddings, int height = 24, int width = 24);

// Merge overlapping patches back into a single feature map.
// Input: [num_patches * batch_size, H, W, C] in NHWC
// Returns: [batch_size, merged_H, merged_W, C]
mx::array merge_patches(const mx::array& x, int batch_size, int steps, int padding);

// --- Encoder post-ViT ---
struct EncoderPost {
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::unordered_map<std::string, mx::array>& all_weights);

    // Takes ViT outputs, returns 5 feature maps in NHWC layout.
    // Each element: [1, H, W, C]
    std::vector<mx::array> forward(
        const mx::array& patch_enc_out,   // [35, 577, 1024]
        const mx::array& hook0_out,       // [35, 577, 1024]
        const mx::array& hook1_out,       // [35, 577, 1024]
        const mx::array& image_enc_out,   // [1, 577, 1024]
        int batch_size = 1);
};

// --- Decoder ---
struct Decoder {
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::unordered_map<std::string, mx::array>& all_weights);

    // Takes 5 encoder features (NHWC), returns (features, lowres_features).
    std::pair<mx::array, mx::array> forward(const std::vector<mx::array>& encodings);
};

// --- Depth Head ---
struct DepthHead {
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::unordered_map<std::string, mx::array>& all_weights);

    // Input: [1, 768, 768, 256] NHWC → Output: [1, 1536, 1536, 1]
    mx::array forward(const mx::array& features);
};

// --- FOV Head ---
struct FOVHead {
    std::unordered_map<std::string, mx::array> weights;

    void load(const std::unordered_map<std::string, mx::array>& all_weights);

    // fov_enc_out: [1, 577, 1024], lowres_feature: [1, 48, 48, 256] NHWC
    // Returns: scalar fov in degrees
    mx::array forward(const mx::array& fov_enc_out, const mx::array& lowres_feature);
};

// --- Top-level model ---
struct DepthProModel {
    EncoderPost encoder;
    Decoder decoder;
    DepthHead head;
    FOVHead fov;

    void load(const std::string& safetensors_path);

    // Full forward: takes 5 ViT outputs, returns (depth, fov_deg).
    std::pair<mx::array, mx::array> forward(
        const mx::array& patch_enc_out,
        const mx::array& hook0_out,
        const mx::array& hook1_out,
        const mx::array& image_enc_out,
        const mx::array& fov_enc_out);
};

} // namespace depth_pro
