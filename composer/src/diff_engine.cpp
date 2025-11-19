#include "diff_engine.hpp"
#include <cmath>

std::vector<Order> DiffEngine::calculate_orders(
    const std::map<std::string, int>& target_quantities,
    const Portfolio& current_portfolio
) {
    std::vector<Order> orders;

    // 1. Handle Buys and Sells for assets in Target
    for (const auto& [symbol, target_qty] : target_quantities) {
        int current_qty = 0;
        if (current_portfolio.positions.count(symbol)) {
            current_qty = current_portfolio.positions.at(symbol).quantity;
        }

        int diff = target_qty - current_qty;

        if (diff > 0) {
            orders.push_back({symbol, Side::BUY, diff, OrderType::MARKET, std::nullopt});
        } else if (diff < 0) {
            orders.push_back({symbol, Side::SELL, std::abs(diff), OrderType::MARKET, std::nullopt});
        }
    }

    // 2. Handle Liquidations (Assets in Current but NOT in Target)
    for (const auto& [symbol, position] : current_portfolio.positions) {
        if (target_quantities.find(symbol) == target_quantities.end()) {
            // Asset exists in portfolio but not in target -> Liquidate all
            if (position.quantity > 0) {
                orders.push_back({symbol, Side::SELL, position.quantity, OrderType::MARKET, std::nullopt});
            }
        }
    }

    return orders;
}
