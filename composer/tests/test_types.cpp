#include <gtest/gtest.h>
#include "types.hpp"
#include <nlohmann/json.hpp>

TEST(TypesTest, OrderSerialization) {
    Order order{"AAPL", Side::BUY, 100, OrderType::MARKET, std::nullopt};
    nlohmann::json j = order;
    
    EXPECT_EQ(j["symbol"], "AAPL");
    EXPECT_EQ(j["side"], "BUY");
    EXPECT_EQ(j["quantity"], 100);
    EXPECT_EQ(j["type"], "MARKET");
}

TEST(TypesTest, PortfolioSerialization) {
    Portfolio p;
    p.cash = 10000.0;
    p.positions["AAPL"] = Position{"AAPL", 10, 150.0};

    nlohmann::json j = p;
    EXPECT_EQ(j["cash"], 10000.0);
    EXPECT_EQ(j["positions"]["AAPL"]["symbol"], "AAPL");
}
