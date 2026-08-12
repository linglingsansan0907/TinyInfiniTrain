#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    // 数据集文件（例如 tinyshakespeare.bin,里的真实token数量小于Header 里声明的 num_toks（或者文件读到了 EOF 结尾，实际剩余字节根本没有 total_data_bytes 那么长）


    // 以二进制只读方式打开数据集文件
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    // 读1024字节的header头部
    std::vector<uint8_t> header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    // 解析header
    int magic = BytesToType<int>(header_bytes, 0);  // 0-3,说明文件格式类型 GPT-2格式等
    int version = BytesToType<int>(header_bytes, 4); // 4-7, 数据文件版本号
    int num_toks = BytesToType<int>(header_bytes, 8); // 8-11, 文件里面存储多少token
    // 校验magic字段，并在kTypeMap中获取token类型
    CHECK(kTypeMap.find(magic) != kTypeMap.end()) << "Invalid magic number: " << magic;
    TinyShakespeareType token_type = kTypeMap.at(magic);
    size_t token_bytes = kTypeToSize.at(token_type);
    DataType data_type = kTypeToDataType.at(token_type);


    // 获取文件剩余实际大小,防止num_toks越界
    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    size_t actual_data_bytes = (file_size > 1024) ? (file_size - 1024) : 0;
    ifs.seekg(1024, std::ios::beg); // 回到DATA起始位置

    // 计算实际可用的total_data_bytes
    size_t expected_data_bytes = static_cast<size_t>(num_toks) * token_bytes;
    size_t total_data_bytes = std::min(expected_data_bytes, actual_data_bytes);

    // 安全读取data_bytes
    std::vector<uint8_t> data_bytes(total_data_bytes);
    ifs.read(reinterpret_cast<char *>(data_bytes.data()), total_data_bytes);
    // // 读取后续的DATA(Token ID)
    // size_t total_data_bytes = static_cast<size_t>(num_toks) * token_bytes;
    // std::vector<uint8_t> data_bytes = ReadSeveralBytesFromIfstream(total_data_bytes, &ifs);
    size_t actual_num_toks = total_data_bytes / token_bytes;


    // 封装为C++结构体 构造TinyShakespeareFile并返回
    TinyShakespeareFile file;
    file.type = token_type;
    // file.magic = magic;
    // file.version = version;
    // file.num_toks = num_toks;
    // file.token_type = token_type;

    // 训练固定sequence_length长度
    int64_t num_samples = actual_num_toks / sequence_length; // 多少个
    file.dims = {num_samples, static_cast<int64_t>(sequence_length)};

    // 创建存储整个token数据的tensor
    // 直接构造tensor，防止超过重载范围
    // Tensor只有3个构造参数！！
    file.tensor = infini_train::Tensor(
        file.dims, DataType::kINT64, infini_train::Device(infini_train::DeviceType::kCPU, 0)
        // data_bytes.data()
    );

    // 二进制uint16_t->int64_t 填入tensor
    int64_t *dst_ptr = static_cast<int64_t *>(file.tensor.DataPtr());
    size_t total_elements = num_samples * sequence_length;
    if (token_type == TinyShakespeareType::kUINT16) {
        const uint16_t *src_ptr = reinterpret_cast<const uint16_t *>(data_bytes.data());
        for (size_t i = 0; i < total_elements; ++i) {
            dst_ptr[i] = static_cast<int64_t>(src_ptr[i]);
        }
    } else if (token_type == TinyShakespeareType::kUINT32) {
        const uint32_t *src_ptr = reinterpret_cast<const uint32_t *>(data_bytes.data());
        for (size_t i = 0; i < total_elements; ++i) {
            dst_ptr[i] = static_cast<int64_t>(src_ptr[i]);
        }
    }
    // // 将二进制文件读出的data_bytes单独复制到tensor内存中
    // size_t copy_bytes = std::min(total_data_bytes, file.tensor.SizeInBytes());
    // std::memcpy(file.tensor.DataPtr(), data_bytes.data(), copy_bytes);
    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    ///// 改用构造函数初始化列表（.h是写死的const ）
    // 让dataset内部其他函数后续也能使用sequence_length
  : sequence_length_(sequence_length),
    text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),

    // 计算单个sequence占用的内存字节大小: 如果kUINIT16: token_bytes 2字节，kUINT32：token_bytes 4字节
    // size_t token_bytes = kTypeToSize.at(text_file_.type);
    sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),

    // 计算总样本数：有效样本总数防止越界而减1
    num_samples_(text_file_.dims[0] > 0 ? text_file_.dims[0] - 1 : 0) {}


std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    
    size_t offset_x = idx * sequence_size_in_bytes_;
    size_t offset_y = offset_x + sizeof(int64_t);

    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, offset_x, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, offset_y, dims)};
    
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    // return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
    //         std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
    //                                                dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
