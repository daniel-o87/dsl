rc// types.h
#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <vector>
#include <set>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// tags for union
typedef enum {
    TYPE_BOOLEAN,
    TYPE_INTEGER,
    TYPE_INTEGER_TUPLE,
    TYPE_GRID
} ElementType;

// tuple structure
typedef struct {
    int32_t x;
    int32_t y;
} IntegerTuple;

// grid structure
typedef struct {
    std::vector<std::vector<int32_t>> data;
    size_t rows;
    size_t cols;
} Grid;

// union for numerical types
typedef struct {
    ElementType type;
    union {
        bool boolean_val;
        int32_t integer_val;
        IntegerTuple tuple_val;
        Grid* grid_val;
    } value;
} Element;

Grid* json_to_grid(const json& grid_json);
void print_grid(const Grid* grid);
void free_grid(Grid* grid);

#endif // TYPES_H
