#include <gtest/gtest.h>
#include "diff_engine.hpp"

TEST(DiffEngineTest, BuyNewPosition) {
    std::map<std::string, int> target = {{"AAPL", 10}};
    Portfolio current; // Empty portfolio
    
    auto orders = DiffEngine::calculate_orders(target, current);
    
    ASSERT_EQ(orders.size(), 1);
    EXPECT_EQ(orders[0].symbol, "AAPL");
    EXPECT_EQ(orders[0].side, Side::BUY);
    EXPECT_EQ(orders[0].quantity, 10);
}

TEST(DiffEngineTest, SellExistingPosition) {
    std::map<std::string, int> target = {{"MSFT", 50}}; // Target 50
    Portfolio current;
    current.positions["MSFT"] = {"MSFT", 100, 250.0}; // Current 100
    
    auto orders = DiffEngine::calculate_orders(target, current);
    
    ASSERT_EQ(orders.size(), 1);
    EXPECT_EQ(orders[0].symbol, "MSFT");
    EXPECT_EQ(orders[0].side, Side::SELL);
    EXPECT_EQ(orders[0].quantity, 50);
}

TEST(DiffEngineTest, LiquidatePosition) {
    std::map<std::string, int> target; // Empty target
    Portfolio current;
    current.positions["TSLA"] = {"TSLA", 20, 800.0};
    
    auto orders = DiffEngine::calculate_orders(target, current);
    
    ASSERT_EQ(orders.size(), 1);
    EXPECT_EQ(orders[0].symbol, "TSLA");
    EXPECT_EQ(orders[0].side, Side::SELL);
    EXPECT_EQ(orders[0].quantity, 20);
}
