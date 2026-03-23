#include "model.h"
#include <cmath>
#include <stdexcept>

namespace depth_pro {

// --- Utility ops ---

mx::array conv2d_forward(
    const mx::array& input,
    const mx::array& weight,
    std::pair<int,int> stride,
    std::pair<int,int> padding) {
    return mx::conv2d(input, weight, stride, padding);
}

mx::array conv2d_forward(
    const mx::array& input,
    const mx::array& weight,
    const mx::array& bias,
    std::pair<int,int> stride,
    std::pair<int,int> padding) {
    auto out = mx::conv2d(input, weight, stride, padding);
    // bias shape [O] -> broadcast over [N,H,W,O]
    return out + bias;
}

mx::array conv_transpose2d_forward(
    const mx::array& input,
    const mx::array& weight,
    std::pair<int,int> stride,
    std::pair<int,int> padding) {
    return mx::conv_transpose2d(input, weight, stride, padding);
}

mx::array conv_transpose2d_forward(
    const mx::array& input,
    const mx::array& weight,
    const mx::array& bias,
    std::pair<int,int> stride,
    std::pair<int,int> padding) {
    auto out = mx::conv_transpose2d(input, weight, stride, padding);
    return out + bias;
}

mx::array relu(const mx::array& x) {
    return mx::maximum(x, mx::array(0.0f));
}

mx::array reshape_feature(const mx::array& embeddings, int height, int width) {
    // embeddings: [N, 577, 1024] — remove CLS token at position 0
    int n = embeddings.shape(0);
    int c = embeddings.shape(2);
    // Slice [N, 1:, :] to get [N, 576, 1024]
    auto tokens = mx::slice(embeddings, {0, 1, 0}, {n, 1 + height * width, c});
    // Reshape to [N, H, W, C] (already NHWC)
    return mx::reshape(tokens, {n, height, width, c});
}

mx::array merge_patches(const mx::array& x, int batch_size, int steps, int padding) {
    // x: [steps*steps*batch_size, pH, pW, C]
    // Merge into [batch_size, merged_H, merged_W, C]
    int idx = 0;
    std::vector<mx::array> output_rows;

    for (int j = 0; j < steps; j++) {
        std::vector<mx::array> row_patches;
        for (int i = 0; i < steps; i++) {
            int start = batch_size * idx;
            int end = batch_size * (idx + 1);
            // Slice batch dimension
            auto patch = mx::slice(x, {start, 0, 0, 0},
                                   {end, x.shape(1), x.shape(2), x.shape(3)});

            int h_start = 0, h_end = patch.shape(1);
            int w_start = 0, w_end = patch.shape(2);

            if (j != 0) h_start = padding;
            if (j != steps - 1) h_end -= padding;
            if (i != 0) w_start = padding;
            if (i != steps - 1) w_end -= padding;

            patch = mx::slice(patch,
                              {0, h_start, w_start, 0},
                              {patch.shape(0), h_end, w_end, patch.shape(3)});

            row_patches.push_back(patch);
            idx++;
        }
        // Concatenate along width (axis=2)
        auto row = mx::concatenate(row_patches, 2);
        output_rows.push_back(row);
    }
    // Concatenate along height (axis=1)
    return mx::concatenate(output_rows, 1);
}

// --- Sequential Conv + ConvTranspose upsample block ---
// Runs: Conv2d(1x1, no bias) -> N x ConvTranspose2d(k2,s2, no bias)
static mx::array run_upsample_block(
    const mx::array& input,
    const std::unordered_map<std::string, mx::array>& w,
    const std::string& prefix,
    int num_convt) {
    // First layer: 1x1 conv projection
    auto x = conv2d_forward(input, get_weight(w, prefix + ".0.weight"));
    // ConvTranspose2d layers
    for (int i = 0; i < num_convt; i++) {
        auto key = prefix + "." + std::to_string(i + 1) + ".weight";
        x = conv_transpose2d_forward(x, get_weight(w, key));
    }
    return x;
}

// --- ResidualBlock ---
// residual = ReLU -> Conv -> ReLU -> Conv
// output = input + residual
static mx::array residual_block_forward(
    const mx::array& input,
    const mx::array& conv1_w,
    const mx::array& conv1_b,
    const mx::array& conv2_w,
    const mx::array& conv2_b) {
    auto x = relu(input);
    x = conv2d_forward(x, conv1_w, conv1_b, {1,1}, {1,1});
    x = relu(x);
    x = conv2d_forward(x, conv2_w, conv2_b, {1,1}, {1,1});
    return input + x;
}

// --- FeatureFusionBlock2d ---
static mx::array fusion_block_forward(
    const mx::array& x0,
    const mx::array* x1,  // nullable
    const std::unordered_map<std::string, mx::array>& w,
    const std::string& prefix,
    bool use_deconv) {

    auto x = x0;

    if (x1 != nullptr) {
        // resnet1 on x1
        auto res = residual_block_forward(
            *x1,
            get_weight(w, prefix + ".resnet1.residual.1.weight"),
            get_weight(w, prefix + ".resnet1.residual.1.bias"),
            get_weight(w, prefix + ".resnet1.residual.3.weight"),
            get_weight(w, prefix + ".resnet1.residual.3.bias"));
        x = x + res;
    }

    // resnet2
    x = residual_block_forward(
        x,
        get_weight(w, prefix + ".resnet2.residual.1.weight"),
        get_weight(w, prefix + ".resnet2.residual.1.bias"),
        get_weight(w, prefix + ".resnet2.residual.3.weight"),
        get_weight(w, prefix + ".resnet2.residual.3.bias"));

    if (use_deconv) {
        x = conv_transpose2d_forward(x, get_weight(w, prefix + ".deconv.weight"));
    }

    x = conv2d_forward(x,
                        get_weight(w, prefix + ".out_conv.weight"),
                        get_weight(w, prefix + ".out_conv.bias"));
    return x;
}

// --- EncoderPost ---

void EncoderPost::load(const std::unordered_map<std::string, mx::array>& all_weights) {
    for (auto& [k, v] : all_weights) {
        if (k.find("encoder.upsample") == 0 || k.find("encoder.fuse") == 0) {
            weights.insert_or_assign(k, v);
        }
    }
}

std::vector<mx::array> EncoderPost::forward(
    const mx::array& patch_enc_out,
    const mx::array& hook0_out,
    const mx::array& hook1_out,
    const mx::array& image_enc_out,
    int batch_size) {

    // Reshape all ViT outputs: remove CLS, reshape to [N, 24, 24, C] NHWC
    auto x_pyramid = reshape_feature(patch_enc_out);    // [35, 24, 24, 1024]
    auto x_hook0 = reshape_feature(hook0_out);           // [35, 24, 24, 1024]
    auto x_hook1 = reshape_feature(hook1_out);           // [35, 24, 24, 1024]
    auto x_global = reshape_feature(image_enc_out);      // [1, 24, 24, 1024]

    // Merge hook features (5x5 grid, padding=3)
    auto x_hook0_25 = mx::slice(x_hook0, {0,0,0,0}, {batch_size * 25, 24, 24, 1024});
    auto x_latent0 = merge_patches(x_hook0_25, batch_size, 5, 3); // [1, 96, 96, 1024]

    auto x_hook1_25 = mx::slice(x_hook1, {0,0,0,0}, {batch_size * 25, 24, 24, 1024});
    auto x_latent1 = merge_patches(x_hook1_25, batch_size, 5, 3); // [1, 96, 96, 1024]

    // Split pyramid encodings: first 25, next 9, last 1
    auto x0_enc = mx::slice(x_pyramid, {0,0,0,0}, {batch_size * 25, 24, 24, 1024});
    auto x1_enc = mx::slice(x_pyramid, {batch_size * 25, 0, 0, 0},
                            {batch_size * 34, 24, 24, 1024});
    auto x2_enc = mx::slice(x_pyramid, {batch_size * 34, 0, 0, 0},
                            {batch_size * 35, 24, 24, 1024});

    // Merge patches
    auto x0_features = merge_patches(x0_enc, batch_size, 5, 3);  // [1, 96, 96, 1024]
    auto x1_features = merge_patches(x1_enc, batch_size, 3, 6);  // [1, 48, 48, 1024]
    auto x2_features = x2_enc;                                     // [1, 24, 24, 1024]

    // Upsample blocks
    // upsample_latent0: Conv(1024->256, k1) + 3x ConvT(s2)
    x_latent0 = run_upsample_block(x_latent0, weights, "encoder.upsample_latent0", 3);
    // -> [1, 768, 768, 256]

    // upsample_latent1: Conv(1024->256, k1) + 2x ConvT(s2)
    x_latent1 = run_upsample_block(x_latent1, weights, "encoder.upsample_latent1", 2);
    // -> [1, 384, 384, 256]

    // upsample0: Conv(1024->512, k1) + ConvT(s2)
    x0_features = run_upsample_block(x0_features, weights, "encoder.upsample0", 1);
    // -> [1, 192, 192, 512]

    // upsample1: Conv(1024->1024, k1) + ConvT(s2)
    x1_features = run_upsample_block(x1_features, weights, "encoder.upsample1", 1);
    // -> [1, 96, 96, 1024]

    // upsample2: Conv(1024->1024, k1) + ConvT(s2)
    x2_features = run_upsample_block(x2_features, weights, "encoder.upsample2", 1);
    // -> [1, 48, 48, 1024]

    // upsample_lowres (ConvTranspose2d with bias)
    x_global = conv_transpose2d_forward(
        x_global,
        get_weight(weights, "encoder.upsample_lowres.weight"),
        get_weight(weights, "encoder.upsample_lowres.bias"));
    // -> [1, 48, 48, 1024]

    // fuse_lowres: cat along channel + Conv(2048->1024, k1)
    auto fused = mx::concatenate({x2_features, x_global}, 3); // [1, 48, 48, 2048]
    auto x_global_fused = conv2d_forward(
        fused,
        get_weight(weights, "encoder.fuse_lowres.weight"),
        get_weight(weights, "encoder.fuse_lowres.bias"));
    // -> [1, 48, 48, 1024]

    // Return 5 feature maps (same order as PyTorch):
    // [0] x_latent0: [1, 768, 768, 256]
    // [1] x_latent1: [1, 384, 384, 256]
    // [2] x0:        [1, 192, 192, 512]
    // [3] x1:        [1, 96, 96, 1024]
    // [4] x_global:  [1, 48, 48, 1024]
    return {x_latent0, x_latent1, x0_features, x1_features, x_global_fused};
}

// --- Decoder ---

void Decoder::load(const std::unordered_map<std::string, mx::array>& all_weights) {
    for (auto& [k, v] : all_weights) {
        if (k.find("decoder.") == 0) {
            weights.insert_or_assign(k, v);
        }
    }
}

std::pair<mx::array, mx::array> Decoder::forward(const std::vector<mx::array>& encodings) {
    // Project all features to decoder dim (256).
    // convs[0] is Identity when dims match (256==256), otherwise 1x1 conv.
    // We check if the weight exists; if not, it's Identity.
    std::vector<mx::array> projected;
    projected.reserve(encodings.size());

    for (int i = 0; i < (int)encodings.size(); i++) {
        auto key = "decoder.convs." + std::to_string(i) + ".weight";
        auto it = weights.find(key);
        if (it != weights.end()) {
            projected.push_back(conv2d_forward(encodings[i], it->second, {1,1},
                                               i == 0 ? std::make_pair(0,0) : std::make_pair(1,1)));
        } else {
            projected.push_back(encodings[i]);
        }
    }

    int num_levels = (int)projected.size();

    // Start from lowest resolution
    auto features = projected[num_levels - 1];
    auto lowres_features = features;

    // fusion[num_levels-1]: no skip input
    features = fusion_block_forward(
        features, nullptr, weights,
        "decoder.fusions." + std::to_string(num_levels - 1),
        true);  // use_deconv for all except fusion[0]

    // Fuse from low to high resolution
    for (int i = num_levels - 2; i >= 0; i--) {
        auto& feat_i = projected[i];
        features = fusion_block_forward(
            features, &feat_i, weights,
            "decoder.fusions." + std::to_string(i),
            i != 0);  // no deconv at level 0
    }

    return {features, lowres_features};
}

// --- DepthHead ---

void DepthHead::load(const std::unordered_map<std::string, mx::array>& all_weights) {
    for (auto& [k, v] : all_weights) {
        if (k.find("head.") == 0) {
            weights.insert_or_assign(k, v);
        }
    }
}

mx::array DepthHead::forward(const mx::array& features) {
    // head[0]: Conv2d(256->128, k3, p1)
    auto x = conv2d_forward(features,
                             get_weight(weights, "head.0.weight"),
                             get_weight(weights, "head.0.bias"),
                             {1,1}, {1,1});
    // head[1]: ConvTranspose2d(128->128, k2, s2)
    x = conv_transpose2d_forward(x,
                                  get_weight(weights, "head.1.weight"),
                                  get_weight(weights, "head.1.bias"));
    // head[2]: Conv2d(128->32, k3, p1)
    x = conv2d_forward(x,
                        get_weight(weights, "head.2.weight"),
                        get_weight(weights, "head.2.bias"),
                        {1,1}, {1,1});
    // head[3]: ReLU
    x = relu(x);
    // head[4]: Conv2d(32->1, k1)
    x = conv2d_forward(x,
                        get_weight(weights, "head.4.weight"),
                        get_weight(weights, "head.4.bias"));
    // head[5]: ReLU
    x = relu(x);
    return x;
}

// --- FOVHead ---

void FOVHead::load(const std::unordered_map<std::string, mx::array>& all_weights) {
    for (auto& [k, v] : all_weights) {
        // fov.encoder.1.* is the Linear layer (fov.encoder.0 is the ViT, stays in PyTorch)
        // fov.downsample.* and fov.head.* are conv layers
        if (k.find("fov.") == 0 &&
            k.find("fov.encoder.0.") != 0) {
            weights.insert_or_assign(k, v);
        }
    }
}

mx::array FOVHead::forward(const mx::array& fov_enc_out, const mx::array& lowres_feature) {
    // fov_enc_out: [1, 577, 1024] from the ViT

    // Linear projection: [1, 577, 1024] -> remove CLS -> [1, 576, 1024]
    int n = fov_enc_out.shape(0);
    int c = fov_enc_out.shape(2);
    auto tokens = mx::slice(fov_enc_out, {0, 1, 0}, {n, 577, c}); // [1, 576, 1024]

    // Apply Linear(1024 -> num_features/2 = 128)
    auto linear_w = get_weight(weights, "fov.encoder.1.weight"); // [128, 1024]
    auto linear_b = get_weight(weights, "fov.encoder.1.bias");   // [128]
    // matmul: [1, 576, 1024] @ [1024, 128] + [128]
    auto x = mx::addmm(linear_b, tokens, mx::transpose(linear_w));
    // x: [1, 576, 128]

    // Reshape to [1, 24, 24, 128] (NHWC)
    x = mx::reshape(x, {1, 24, 24, 128});

    // Downsample lowres_feature: Conv2d(256->128, k3, s2, p1) + ReLU
    auto lowres = conv2d_forward(
        lowres_feature,
        get_weight(weights, "fov.downsample.0.weight"),
        get_weight(weights, "fov.downsample.0.bias"),
        {2,2}, {1,1});
    lowres = relu(lowres);
    // lowres: [1, 24, 24, 128]

    // Add
    x = x + lowres;

    // FOV head convolutions
    // head[0]: Conv2d(128->64, k3, s2, p1)
    x = conv2d_forward(x,
                        get_weight(weights, "fov.head.0.weight"),
                        get_weight(weights, "fov.head.0.bias"),
                        {2,2}, {1,1});
    // head[1]: ReLU
    x = relu(x);
    // head[2]: Conv2d(64->32, k3, s2, p1)
    x = conv2d_forward(x,
                        get_weight(weights, "fov.head.2.weight"),
                        get_weight(weights, "fov.head.2.bias"),
                        {2,2}, {1,1});
    // head[3]: ReLU
    x = relu(x);
    // head[4]: Conv2d(32->1, k6, s1, p0)
    x = conv2d_forward(x,
                        get_weight(weights, "fov.head.4.weight"),
                        get_weight(weights, "fov.head.4.bias"));
    // x: [1, 1, 1, 1]
    return x;
}

// --- DepthProModel ---

void DepthProModel::load(const std::string& safetensors_path) {
    auto [all_weights, metadata] = mx::load_safetensors(safetensors_path);
    encoder.load(all_weights);
    decoder.load(all_weights);
    head.load(all_weights);
    fov.load(all_weights);
}

std::pair<mx::array, mx::array> DepthProModel::forward(
    const mx::array& patch_enc_out,
    const mx::array& hook0_out,
    const mx::array& hook1_out,
    const mx::array& image_enc_out,
    const mx::array& fov_enc_out) {

    // Encoder post-ViT
    auto features = encoder.forward(patch_enc_out, hook0_out, hook1_out, image_enc_out);

    // Decoder
    auto [decoded, lowres_features] = decoder.forward(features);

    // Depth head
    auto depth = head.forward(decoded);

    // FOV head
    auto fov_deg = fov.forward(fov_enc_out, lowres_features);

    return {depth, fov_deg};
}

} // namespace depth_pro
