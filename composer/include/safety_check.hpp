#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <optional>

class SafetyCheck {
public:
    struct Result {
        bool valid;
        std::string reason;
    };

    static Result validate(const std::vector<Order>& orders);
};
