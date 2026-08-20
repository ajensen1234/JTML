#pragma once
#include <functional>
#include "domain/data_structures_6D.h"

namespace throughput_serial {
struct SerialContext {
    void* pipeline = nullptr; // opaque Pipeline*
    std::function<double(const Point6D&)> cost;
    ~SerialContext();
    SerialContext(SerialContext&&) noexcept;
    SerialContext& operator=(SerialContext&&) noexcept;
    SerialContext(const SerialContext&) = delete;
    SerialContext& operator=(const SerialContext&) = delete;
    explicit SerialContext(void* p, std::function<double(const Point6D&)> c) : pipeline(p), cost(std::move(c)) {}
};

SerialContext CreateSerialContext();
}
