#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    // 开文件，读header(1024字节)
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;
    std::vector<uint8_t> header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);

    // header数据
    magic_number_ = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    vocab_size_ = BytesToType<uint32_t>(header_bytes, 8);

    // End of Text token 
    if (kEotMap.find(magic_number_) != kEotMap.end()) {
        eot_token_ = kEotMap.at(magic_number_);
    }

    // 读取填充vocab table
    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        // 获取长度
        std::vector<uint8_t> len_byte = ReadSeveralBytesFromIfstream(1, &ifs);
        uint8_t len = len_byte[0];

        // 获得字符内容
        std::vector<uint8_t> str_bytes = ReadSeveralBytesFromIfstream(len, &ifs);
        token_table_[i] = std::string(str_bytes.begin(), str_bytes.end());
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    if (token_id < token_table_.size()) { // 索引是否在范围内
        return token_table_[token_id];
    }
    return "";
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
#ifndef USE_CUDA
    device = Device(DeviceType::kCPU, 0);
#endif 

    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t rng_state = kRngState;

    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        
        auto x_dev = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        // forward,把有已生成序列的x送入GPT-2，计算logits
        auto outputs = model.Forward({x}); // [B, sequence_length, vocab_size]
        // 把推理得到的logits放回CPU，在CPU上Softmax和采样计算
        // Softmax和计算的kernel不好写，而计算量又比较小，所以就用CPU
        auto logits = outputs[0]->To(Device(DeviceType::kCPU, 0));

        // 预测logits数据,并计算偏移量,定位t-1位置的启始指针位置
        const float *logits_ptr = static_cast<const float *>(logits.DataPtr());
        int64_t offset = (0 * sequence_length + (t - 1)) * vocab_size_; // 0是因为是单文本生成
        const float *tok_logits = logits_ptr + offset; // 第0个batch, t-1个token对应的数值的起点处

        // 计算softmax 把数值转换为概率分布
        std::vector<float> probs(vocab_size_);
        // 寻找logits中的最大值 防止NaN
        float max_logit = tok_logits[0];
        for (uint32_t v = 1; v < vocab_size_; ++v) {
            if (tok_logits[v] > max_logit) {
                max_logit = tok_logits[v];
            }
        }

        // 计算exp (logit - max_logit)并求和
        float sum_exp = 0.0f;
        for (uint32_t v = 0; v < vocab_size_; ++v) {
            probs[v] = std::exp(tok_logits[v] - max_logit);
            sum_exp += probs[v];
        }
        // 归一化
        for (uint32_t v = 0; v < vocab_size_; ++v) {
            probs[v] /= sum_exp;
        }

        //浮点数
        float coin = RandomF32(rng_state);
        // 采样, 并提取预测的token ID
        int next_token = SampleMult(probs.data(), vocab_size_, coin);

        // 控制台输出
        std::string piece = Decode(next_token);
        std::cout << piece << std::flush;
        
        // 更新tensor，自回归迭代
        // next_token也写入CPU缓冲区的第t个位置

        // if (t < sequence_length) {
        //     int64_t *x_cpu_data = static_cast<int64_t *>(x_tensor.DataPtr());
        //     x_cpu_data[t] = next_token;
        //     x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        // }
        if (t < sequence_length) {
            x_buff[t] = next_token;
            // x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        }

        // if (device.Type() == infini_train::DeviceType::kCUDA) {
        //     // 如果 x 是 CUDA Tensor，先转回到 CPU 修改或直接更新 CPU 副本
        //     auto x_cpu = x->To(infini_train::Device(infini_train::DeviceType::kCPU, 0));
        //     int64_t *x_ptr = static_cast<int64_t *>(x_cpu.DataPtr());
        //     if (t + 1 < sequence_length) {
        //         x_ptr[t + 1] = next_token;
        //     }
        //     x = std::make_shared<infini_train::Tensor>(x_cpu.To(device));
        // } else {
        //     int64_t *x_ptr = static_cast<int64_t *>(x->DataPtr());
        //     if (t + 1 < sequence_length) {
        //         x_ptr[t + 1] = next_token;
        //     }
        // }


    }
    std::cout << std::endl;
}
} // namespace infini_train
