/* Plan 012 U1: GraphAdmissionPolicy default deny (C10,R14). CUDA-free. */
#pragma once

#include <string>

namespace gpu_cost_function {

struct GraphAdmissionEvidence {
    bool runtimeOptIn = false;
    int layeredArtifactVersion = 0;
    bool throughputRetained = false;
};

class GraphAdmissionPolicy {
public:
    virtual ~GraphAdmissionPolicy() = default;
    virtual bool admit(const GraphAdmissionEvidence& e) const {
        return e.runtimeOptIn && e.layeredArtifactVersion >= 1 &&
            e.throughputRetained;
    }
    virtual std::string denyReason(const GraphAdmissionEvidence& e) const {
        if (!e.runtimeOptIn) {
            return "runtimeOptIn not set";
        }
        if (e.layeredArtifactVersion < 1) {
            return "layeredArtifactVersion < 1";
        }
        if (!e.throughputRetained) {
            return "throughputRetained not set";
        }
        return "";
    }
};

struct GraphAdmissionInputs {
    bool executorReady = false;
    bool monoplaneEligible = false;
    bool recipeFound = false;
    bool preflightCapturable = false;
    GraphAdmissionEvidence evidence;
};

struct GraphAdmissionDecision {
    bool install = false;
    std::string reason;
};

inline GraphAdmissionDecision DecideGraphAdmission(
    const GraphAdmissionInputs& in,
    const GraphAdmissionPolicy& policy) {
    if (!in.executorReady) {
        return {false, "executor not ready"};
    }
    if (!in.monoplaneEligible) {
        return {false, "not monoplane DIRECT_DILATION eligible"};
    }
    if (!in.recipeFound) {
        return {false, "no eligible recipe"};
    }
    if (!in.preflightCapturable) {
        return {false, "preflight not capturable"};
    }
    if (!policy.admit(in.evidence)) {
        std::string r = policy.denyReason(in.evidence);
        if (r.empty()) {
            r = "policy denied";
        }
        return {false, r};
    }
    return {true, ""};
}

}  // namespace gpu_cost_function
