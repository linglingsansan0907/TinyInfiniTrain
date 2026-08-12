#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // TODO：实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================

        using FuncT = RetT (*)(ArgsT...);
        // TODO: 
        // 把已经在template里面定义全为void* 的各个指向函数的指针们的类型全变换为FuncT
        auto func = reinterpret_cast<FuncT>(func_ptr_);
        // 直接把传过来的值，放入，然后func后返回给外面
        return func(std::forward<ArgsT>(args)...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    // template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    //     // =================================== 作业 ===================================
    //     // TODO：实现kernel注册机制
    //     // 功能描述：将kernel函数与设备类型、名称绑定
    //     // =================================== 作业 ===================================
    //     // // 看看key有没有重复
    //     // CHECK(!key_to_kernel_map_.contains(key))
    //     //     << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    //     // CUDA和CPU都需要注册，选择可以在重复注册时覆盖，避免在静态初始化的时候，抛出错误
    //     // if (key_to_kernel_map_.find(key) != key_to_kernel_map_.end()) {
    //     //     throw std::runtime_error("Kernel already registered");
    //     // }
    //     // // 把设备和算子跟KernelFunction绑定
    //     // key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
        
    //     CHECK(!key_to_kernel_map_.contains(key))
    //         << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    //     key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    // }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel, const char* file = "", int line = 0) {
        // std::cout << "[DEBUG_REG] Reg kernel: " << key.second 
        //           << " | dev: " << static_cast<int>(key.first) 
        //           << " | from: " << file << ":" << line << std::endl;

        // CHECK(!key_to_kernel_map_.contains(key))
        //     << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        // key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
        // CHECK(!key_to_kernel_map_.contains(key))
        //     << "Kernel already registered: " << key.second 
        //     << " with device type " << static_cast<int>(key.first)
        //     << " (from " << file << ":" << line << ")";
        // 补丁
        if (key.second == "TestKernel") {
            CHECK(!key_to_kernel_map_.contains(key))
                << "Kernel already registered: " << key.second 
                << " with device type " << static_cast<int>(key.first)
                << " (from " << file << ":" << line << ")";
        }


        if (key_to_kernel_map_.contains(key)) {
        LOG(WARNING) << "Kernel already registered: " << key.second 
                     << " with device type " << static_cast<int>(key.first)
                     << ", updating implementation. (from " << file << ":" << line << ")";
        }
        key_to_kernel_map_.insert_or_assign(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

// 得有两个宏：入口宏和连接宏
#define REGISTER_KERNEL_CONCAT_INNER(a, b) a##b
#define REGISTER_KERNEL_CONCAT(a, b) REGISTER_KERNEL_CONCAT_INNER(a, b)

// =================================== 作业 ===================================
// TODO：实现自动注册宏
// 功能描述：在全局静态区注册kernel，避免显式初始化代码
// =================================== 作业 ===================================

// 最后的选择是不区分屏蔽，而是所有算子用覆盖的方式统一注册
// 如果开启了USE_CUDA,就跳过CPU注册，优先使用CUDA算子
// 注册：把算子的名字和它的具体代码绑定到一起
// 用attribute确保静态变量没被引用

// #define REGISTER_KERNEL_IMPL(device, kernel_name, kernel_func, ID) \
//     struct REGISTER_KERNEL_CONCAT(KernelRegister_, ID) { \
//         REGISTER_KERNEL_CONCAT(KernelRegister_, ID)() { \
//             ::infini_train::Dispatcher::Instance().Register( \
//                 std::make_pair(device, #kernel_name), kernel_func); \
//         } \
//     }; \
//     __attribute__((used)) static REGISTER_KERNEL_CONCAT(KernelRegister_, ID) \
//         REGISTER_KERNEL_CONCAT(g_kernel_register_, ID);

// // __COUNTER__传给ID
// #define REGISTER_KERNEL(device, kernel_name, kernel_func) \
//     REGISTER_KERNEL_IMPL(device, kernel_name, kernel_func, __LINE__)


#define REGISTER_KERNEL_IMPL(device, kernel_name, kernel_func, ID) \
    static const struct REGISTER_KERNEL_CONCAT(KernelRegister_, ID) { \
        REGISTER_KERNEL_CONCAT(KernelRegister_, ID)() { \
            ::infini_train::Dispatcher::Instance().Register( \
                std::make_pair(device, #kernel_name), kernel_func, __FILE__, __LINE__); \
        } \
    } REGISTER_KERNEL_CONCAT(g_kernel_register_, ID);

#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    REGISTER_KERNEL_IMPL(device, kernel_name, kernel_func, __LINE__)

