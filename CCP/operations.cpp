// operations.cpp
#include "types.h"
#include <iostream>

Grid* json_to_grid(const json& grid_json) {
    try {
        if (!grid_json.is_array() || grid_json.empty() || !grid_json[0].is_array()) {
            std::cerr << "Invalid grid format" << std::endl;
            return nullptr;
        }

        Grid* grid = new Grid;
        grid->data.clear();
        
        for (const auto& row : grid_json) {
            std::vector<int32_t> grid_row;
            for (const auto& cell : row) {
                grid_row.push_back(cell.get<int32_t>());
            }
            grid->data.push_back(grid_row);
        }
        
        grid->rows = grid->data.size();
        grid->cols = grid->data.empty() ? 0 : grid->data[0].size();
        
        return grid;
    } catch (const json::exception& e) {
        std::cerr << "JSON error in json_to_grid: " << e.what() << std::endl;
        return nullptr;
    }
}

void free_grid(Grid* grid) {
    if (grid) {
        grid->data.clear();
        delete grid;
    }
}
