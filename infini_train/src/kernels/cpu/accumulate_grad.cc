#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // 得到参数总量
    const int64_t num_elements = grad->NumElements();
    // 确保可以使用各个传入tensor
    const float *g_data = static_cast<const float *>(grad->DataPtr());
    float *p_data = static_cast<float *>(param->DataPtr());
    float *m_data = static_cast<float *>(m->DataPtr());
    float *v_data = static_cast<float *>(v->DataPtr());

    // 偏差修正
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    // 按各个元素地更新Adam的状态与参数
    for (int64_t i = 0; i < num_elements; ++i) {
        // 当前梯度g
        float g = g_data[i];

        // 更新一阶矩 m = beta1 *m + (1-beta1) * g
        m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * g;
        // 更新二阶矩 v = beta2 *v + (1-beta2) * g^2
        v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * g * g;

        //计算偏差修正后的m_hat和v_hat
        float m_hat = m_data[i] / bias_correction1;
        float v_hat = v_data[i] / bias_correction2;

        // 更新权重参数 p = p - lr *m_hat / (sqrt(v_hat) + eps)
        p_data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
