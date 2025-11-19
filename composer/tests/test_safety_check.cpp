#include <gtest/gtest.h>
#include "safety_check.hpp"

TEST(SafetyCheckTest, ValidOrders) {
    std::vector<Order> orders = {
        {"AAPL", Side::BUY, 10, OrderType::MARKET, std::nullopt},
        {"MSFT", Side::SELL, 5, OrderType::MARKET, std::nullopt}
    };

    auto result = SafetyCheck::validate(orders);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.reason, "");
}

TEST(SafetyCheckTest, InvalidQuantityZero) {
    std::vector<Order> orders = {
        {"AAPL", Side::BUY, 0, OrderType::MARKET, std::nullopt}
    };

    auto result = SafetyCheck::validate(orders);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.reason.find("positive"), std::string::npos);
}

TEST(SafetyCheckTest, InvalidQuantityNegative) {
    std::vector<Order> orders = {
        {"AAPL", Side::BUY, -5, OrderType::MARKET, std::nullopt}
    };

    auto result = SafetyCheck::validate(orders);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.reason.find("positive"), std::string::npos);
}

TEST(SafetyCheckTest, InvalidSymbol) {
    std::vector<Order> orders = {
        {"", Side::BUY, 10, OrderType::MARKET, std::nullopt}
    };

    auto result = SafetyCheck::validate(orders);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.reason.find("empty"), std::string::npos);
}
