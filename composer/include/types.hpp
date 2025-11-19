#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <nlohmann/json.hpp>

enum class Side {
    BUY,
    SELL
};

enum class OrderType {
    MARKET,
    LIMIT
};

// Add nlohmann::json serializer for std::optional
namespace nlohmann {
    template <typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& j, const std::optional<T>& opt) {
            if (opt == std::nullopt) {
                j = nullptr;
            } else {
                j = *opt;
            }
        }

        static void from_json(const json& j, std::optional<T>& opt) {
            if (j.is_null()) {
                opt = std::nullopt;
            } else {
                opt = j.get<T>();
            }
        }
    };
}

struct Order {
    std::string symbol;
    Side side;
    int quantity;
    OrderType type;
    std::optional<double> price;

    // Serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Order, symbol, side, quantity, type, price)
};

struct Position {
    std::string symbol;
    int quantity;
    double average_cost;

    // Serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Position, symbol, quantity, average_cost)
};

struct Portfolio {
    std::map<std::string, Position> positions;
    double cash;

    // Serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Portfolio, positions, cash)
};

struct MarketData {
    std::map<std::string, double> prices;

    // Serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(MarketData, prices)
};

// JSON conversions for Enums
NLOHMANN_JSON_SERIALIZE_ENUM(Side, {
    {Side::BUY, "BUY"},
    {Side::SELL, "SELL"}
})

NLOHMANN_JSON_SERIALIZE_ENUM(OrderType, {
    {OrderType::MARKET, "MARKET"},
    {OrderType::LIMIT, "LIMIT"}
})
