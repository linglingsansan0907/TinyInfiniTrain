#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    // 获得input和other的维度 input(B, M, K) other(B, K, N)
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    
    //判断K的维度是不是对的
    const int64_t K = *input_dims.rbegin();
    CHECK_EQ(K, other_dims[other_dims.size() - 2]);

    // 要一个（B， M， N）
    const int64_t N = *other_dims.rbegin();
    const int64_t M = input_dims[input_dims.size() - 2];
    auto output_dims = input_dims;
    *output_dims.rbegin() = N;

    // 用output_dims创建出tensor， 数据类型指定为32位浮点数kFLOAT32
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    // 不能直接Eigen相乘，因为:Eigen会吧-1之前都展平，导致不能匹配
    //直接利用Eigen库重载* 自动在底层调用优化好的CPU矩阵乘法(tensor用eigen转换为矩阵了)
    //output->EigenMatrix() = input->EigenMatrix() * other->EigenMatrix();
    
    const int64_t total_elements = std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>{});
    const int64_t bs = total_elements / (M * K);

    auto A_mat = input->EigenMatrix();
    auto B_mat = other->EigenMatrix();
    auto C_mat = output->EigenMatrix();

    for (int64_t b = 0; b < bs; ++b) {
        C_mat.block(b * M, 0, M, N) = A_mat.block(b * M, 0, M, K) * B_mat.block(b * K, 0, K, N);
    }
    
    //auto output = std::make_shared<Tensor>();
    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = *input_dims.rbegin();
    const int64_t N = *other_dims.rbegin();

    // 存放梯度
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
    
    const int64_t total_elements = std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>{});
    const int64_t bs = total_elements / (M * K);
    // // 矩阵求导
    // // 计算input的梯度 grad_input
    // grad_input->EigenMatrix() = grad_output->EigenMatrix() * other->EigenMatrix().transpose();
    // // 计算other的梯度 grad_other
    // grad_other->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    // 不能默认2D Eigen不能分片，要手动用block分

    auto X_mat = input->EigenMatrix();
    auto W_mat = other->EigenMatrix();
    auto dY_mat = grad_output->EigenMatrix();

    auto dX_mat = grad_input->EigenMatrix();
    auto dW_mat = grad_other->EigenMatrix();

    for (int64_t b = 0; b < bs; ++b){
        // dX = dY * WT
        dX_mat.block(b * M, 0, M, K) = dY_mat.block(b * M, 0, M, N) * W_mat.block(b * K, 0, K, N).transpose();
        // dW = XT * dY
        dW_mat.block(b * K, 0, K, N) = X_mat.block(b * M, 0, M, K).transpose() * dY_mat.block(b * M, 0, M, N);
    }

    // auto grad_input = std::make_shared<Tensor>();
    // auto grad_other = std::make_shared<Tensor>();
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
