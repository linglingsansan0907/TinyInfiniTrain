#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *g, float *p, float *m, float *v,
                                        int64_t num_elements, float learning_rate,
                                        float beta1, float beta2, float eps,
                                        float bias_correction1, float bias_correction2) {

    // 计算当前线程数据索引idx
    // blockIdx.x：block编号；blockDim.x每个block线程数; threadIdx.x当前线程再block内编号
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 防止越界
    if (idx >= num_elements) return;

    // 当前梯度
    float grad = g[idx];

    // 更新一阶矩 m_t = beta1 * m + (1-beta1) *g
    float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
    // 更新二阶矩 v_t = beta2 * v + (1-beta2) * g^2
    float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

    // 把算好的m_val和v_val写回显存，让下一次迭代使用
    m[idx] = m_val;
    v[idx] = v_val;

    // 计算修正后的 m_hat和v_hat
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2; 

    // 更新 p = p -lr *m_hat /(sqrt(v_hat) + eps)
    p[idx] -= learning_rate * m_hat / (sqrt(v_hat) + eps);
                        
}



void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // 所有元素个数
    int64_t num_elements = grad->NumElements();

    // 获取tensor位于GPU的位置
    const float *g_data = static_cast<const float *>(grad->DataPtr());
    float *p_data = static_cast<float *>(param->DataPtr());
    float *m_data = static_cast<float *>(m->DataPtr());
    float *v_data = static_cast<float *>(v->DataPtr());

    // 在CPU中先算好修正系数(1- beta1 ^t) 和(1-beta2 ^t) 这样方便GPU上复用
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    // 配置CUDA的grid网格和block线程块
    int threads_per_block = 256;
    // 覆盖全部元素的block数 (N+255) / 256
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // 启动命令并发
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        g_data, p_data, m_data, v_data,
        num_elements, learning_rate,
        beta1, beta2, eps,
        bias_correction1, bias_correction2

    );

}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
