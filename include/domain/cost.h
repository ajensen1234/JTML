/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * cost.h — the FFI-facing cost handle for the Rust DIRECT port.
 *
 * CppCost seals a std::function<double(const Point6D&)> inside an opaque C++
 * type so the CXX bridge can expose it to Rust as a plain callable
 * (evaluate(Point6D) -> double) without ever passing std::function across the
 * FFI boundary. The concrete std::function is produced by the GPU cost adapter
 * (BuildGpuCostAdapter, coordinator side) and handed to build_cpp_cost();
 * this header stays Qt/GPU-free (it is the domain-purity boundary), so it
 * depends only on <functional> and the Point6D POD.
 *
 * Extern "C++" view (written in the cxx::bridge, not this file):
 *   type CppCost;
 *   fn build_cpp_cost(...) -> UniquePtr<CppCost>;
 *   fn evaluate(self: &CppCost, point: Point6D) -> f64;
 */
#ifndef COST_H
#define COST_H

/*Standard*/
#include <functional>
#include <memory>

/*Header for Point6D*/
#include "domain/data_structures_6D.h"

/*Opaque cost handle: wraps a std::function<double(const Point6D&)> so Rust
  can call it through an opaque type (UniquePtr<CppCost>) + evaluate().*/
class CppCost {
public:
    using CostFunction = std::function<double(const Point6D&)>;

    CppCost() = default;
    explicit CppCost(CostFunction fn) : fn_(std::move(fn)) {}

    /*The score of one pose — the sole call the Rust DIRECT loop makes.*/
    double evaluate(const Point6D& point) const { return fn_(point); }

    /*Whether a cost function has been bound yet (for the default ctor).*/
    bool IsBound() const { return static_cast<bool>(fn_); }

private:
    CostFunction fn_;
};

/*Build the opaque cost handle from a std::function. The concrete function is
  produced upstream (BuildGpuCostAdapter) and boarded here. Rust obtains a
  CppCost by calling this (or via an injected adapter).*/
std::unique_ptr<CppCost> build_cpp_cost(CppCost::CostFunction fn);

#endif /* COST_H */