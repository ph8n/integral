#include "safety_check.hpp"

SafetyCheck::Result SafetyCheck::validate(const std::vector<Order>& orders) {
    for (const auto& order : orders) {
        // Check 1: Quantity > 0
        if (order.quantity <= 0) {
            return {false, "Order quantity must be positive. Found: " + std::to_string(order.quantity) + " for symbol: " + order.symbol};
        }

        // Check 2: Symbol Validity
        if (order.symbol.empty()) {
            return {false, "Order symbol cannot be empty."};
        }
    }

    return {true, ""};
}
