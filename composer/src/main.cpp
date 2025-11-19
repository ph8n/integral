#include <iostream>
#include <vector>
#include <map>
#include "types.hpp"
#include "diff_engine.hpp"
#include "safety_check.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // 1. Read Input (Stdin)
    json input;
    try {
        std::cin >> input;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON input: " << e.what() << std::endl;
        return 1;
    }

    // 2. Parse Target and Current Portfolio
    std::map<std::string, int> target_quantities;
    Portfolio current_portfolio;

    try {
        if (input.contains("target")) {
            target_quantities = input["target"].get<std::map<std::string, int>>();
        }
        if (input.contains("current")) {
            current_portfolio = input["current"].get<Portfolio>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error extracting data from JSON: " << e.what() << std::endl;
        return 1;
    }

    // 3. Run Diff Engine
    std::vector<Order> orders = DiffEngine::calculate_orders(target_quantities, current_portfolio);

    // 4. Run Safety Check
    auto safety_result = SafetyCheck::validate(orders);
    if (!safety_result.valid) {
        std::cerr << "Safety Check Failed: " << safety_result.reason << std::endl;
        
        // Output Error JSON
        json error_output = {
            {"status", "error"},
            {"message", safety_result.reason}
        };
        std::cout << error_output.dump(4) << std::endl;
        return 1;
    }

    // 5. Output Orders (Stdout)
    json output_orders = orders;
    std::cout << output_orders.dump(4) << std::endl;

    return 0;
}
