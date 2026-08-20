/* Plan 012 U1: typed BatchOutcome, not magic vectors (C1,R8/R7). CUDA-free. */
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_cost_function {

struct BatchOutcome {
    enum class Kind {
        NotSubmitted,
        OrderedScores,
        PostLaunchAbort,
        WatchdogPoisoned
    };

    Kind kind = Kind::NotSubmitted;
    std::vector<double> scores;
    std::string reason;

    static BatchOutcome NotSubmitted(std::string reason = {}) {
        BatchOutcome o;
        o.kind = Kind::NotSubmitted;
        o.reason = std::move(reason);
        return o;
    }
    static BatchOutcome Ordered(std::vector<double> scores) {
        BatchOutcome o;
        o.kind = Kind::OrderedScores;
        o.scores = std::move(scores);
        return o;
    }
    static BatchOutcome PostLaunchAbort(std::string reason) {
        BatchOutcome o;
        o.kind = Kind::PostLaunchAbort;
        o.reason = std::move(reason);
        return o;
    }
    static BatchOutcome WatchdogPoisoned(std::string reason) {
        BatchOutcome o;
        o.kind = Kind::WatchdogPoisoned;
        o.reason = std::move(reason);
        return o;
    }

    bool isOrderedScores() const { return kind == Kind::OrderedScores; }
    bool isAbort() const {
        return kind == Kind::PostLaunchAbort || kind == Kind::WatchdogPoisoned;
    }
};

class CoordinatorBatchAbort : public std::runtime_error {
public:
    explicit CoordinatorBatchAbort(BatchOutcome::Kind kind, std::string reason)
        : std::runtime_error(reason.empty() ? "CoordinatorBatchAbort" : reason),
          kind_(kind),
          reason_(std::move(reason)) {}
    BatchOutcome::Kind kind() const noexcept { return kind_; }
    const std::string& reason() const noexcept { return reason_; }

private:
    BatchOutcome::Kind kind_;
    std::string reason_;
};

inline std::vector<double> MaterializeOrderedScores(const BatchOutcome& outcome) {
    if (outcome.kind == BatchOutcome::Kind::OrderedScores) {
        return outcome.scores;
    }
    throw CoordinatorBatchAbort(outcome.kind, outcome.reason);
}

}  // namespace gpu_cost_function
