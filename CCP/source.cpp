// main.cpp
#include "types.h"
#include <iostream>
#include <fstream>

void print_element(const Element& e) {
    switch (e.type) {
        case TYPE_BOOLEAN:
            std::cout << (e.value.boolean_val ? "true" : "false");
            break;
        case TYPE_INTEGER:
            std::cout << e.value.integer_val;
            break;
        case TYPE_INTEGER_TUPLE:
            std::cout << "(" << e.value.tuple_val.x << ", " << e.value.tuple_val.y << ")";
            break;
        case TYPE_GRID:
            if (e.value.grid_val) {
                for (const auto& row : e.value.grid_val->data) {
                    for (int32_t cell : row) {
                        std::cout << cell << " ";
                    }
                    std::cout << "\n";
                }
            }
            break;
    }
}

// Constructor functions for DSL types
Element make_boolean(bool val) {
    Element e;
    e.type = TYPE_BOOLEAN;
    e.value.boolean_val = val;
    return e;
}

Element make_integer(int32_t val) {
    Element e;
    e.type = TYPE_INTEGER;
    e.value.integer_val = val;
    return e;
}

Element make_tuple(int32_t x, int32_t y) {
    Element e;
    e.type = TYPE_INTEGER_TUPLE;
    e.value.tuple_val = {x, y};
    return e;
}

Element make_grid(Grid* grid) {
    Element e;
    e.type = TYPE_GRID;
    e.value.grid_val = grid;
    return e;
}

int main() {
    const std::string challenge_id = "444801d8";
    const std::string train_challenges_path = "./input/arc-agi_training_challenges.json";
    
    try {
        std::ifstream f(train_challenges_path);
        if (!f.is_open()) {
            std::cerr << "Could not open file: " << train_challenges_path << std::endl;
            return 1;
        }
        
        json data = json::parse(f);
        const json& challenge = data[challenge_id];
        const json& train = challenge["train"][0];

        std::cout << "DSL Type Examples:\n";
        
        Element bool_elem = make_boolean(true);
        std::cout << "Boolean: ";
        print_element(bool_elem);
        std::cout << "\n";
        
        Element int_elem = make_integer(42);
        std::cout << "Integer: ";
        print_element(int_elem);
        std::cout << "\n";
        
        Element tuple_elem = make_tuple(3, 4);
        std::cout << "IntegerTuple: ";
        print_element(tuple_elem);
        std::cout << "\n";
        
        std::cout << "\nInput Grid as DSL type:\n";
        Grid* input_grid = json_to_grid(train["input"]);
        Element grid_elem = make_grid(input_grid);
        print_element(grid_elem);
        
        std::cout << "\nOutput Grid as DSL type:\n";
        Grid* output_grid = json_to_grid(train["output"]);
        Element output_grid_elem = make_grid(output_grid);
        print_element(output_grid_elem);
        
        auto demonstrate_numerical = [](const Element& e) {
            if (e.type == TYPE_INTEGER) {
                std::cout << "Numerical (Integer): " << e.value.integer_val << "\n";
            } else if (e.type == TYPE_INTEGER_TUPLE) {
                std::cout << "Numerical (Tuple): (" 
                         << e.value.tuple_val.x << ", " 
                         << e.value.tuple_val.y << ")\n";
            }
        };
        
        std::cout << "\nNumerical type examples:\n";
        demonstrate_numerical(int_elem);     // Integer case
        demonstrate_numerical(tuple_elem);   // IntegerTuple case
        
        free_grid(input_grid);
        free_grid(output_grid);
        
    } catch (const json::exception& e) {
        std::cerr << "JSON error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
