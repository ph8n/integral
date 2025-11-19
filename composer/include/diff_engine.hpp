#pragma once

#include "types.hpp"
#include <vector>
#include <map>

class DiffEngine {
public:
    // Calculate rebalancing orders based on target quantities and current portfolio
    static std::vector<Order> calculate_orders(
        const std::map<std::string, int>& target_quantities,
        const Portfolio& current_portfolio
    );
};
