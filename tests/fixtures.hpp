// Shared test fixtures. Inline MPS text keeps the tests self-contained.
#pragma once

#include <string>

namespace sor::test {

// min  -x1 - 2*x2
// s.t. R1: x1 +   x2 <= 4
//      R2: x1 + 3*x2 <= 6
//      0 <= x1 <= 3, 0 <= x2 <= 3
//
// Vertices of the feasible region: (0,0), (3,0), (3,1), (0,2).
// Objective at (3,1) = -3 - 2 = -5, which is the unique optimum.
inline const std::string kTestLpMps = R"(NAME          TESTLP
ROWS
 N  COST
 L  R1
 L  R2
COLUMNS
    X1        COST      -1.0       R1        1.0
    X1        R2        1.0
    X2        COST      -2.0       R1        1.0
    X2        R2        3.0
RHS
    RHS       R1        4.0        R2        6.0
BOUNDS
 UP BND       X1        3.0
 UP BND       X2        3.0
ENDATA
)";

inline constexpr double kTestLpOptimum = -5.0;
inline constexpr double kTestLpX1 = 3.0;
inline constexpr double kTestLpX2 = 1.0;

// Exercises RANGES, an equality row, MI/FR/FX bounds, an integer MARKER block,
// a second (free) N row, and an objective constant via RHS on the cost row.
inline const std::string kFeaturesMps = R"(NAME          FEATURES
ROWS
 N  OBJ
 N  FREEROW
 E  EQ1
 G  GE1
 L  LE1
COLUMNS
    A         OBJ       1.0        EQ1       1.0
    A         GE1       1.0        FREEROW   99.0
    MARKER                 'MARKER'                 'INTORG'
    B         OBJ       2.0        EQ1       1.0
    B         LE1       1.0
    MARKER                 'MARKER'                 'INTEND'
    C         OBJ       -1.0       LE1       1.0
RHS
    RHS       EQ1       5.0        GE1       1.0
    RHS       LE1       10.0       OBJ       -7.0
RANGES
    RNG       GE1       3.0        LE1       4.0
BOUNDS
 MI BND       A
 FX BND       B         2.0
 FR BND       C
ENDATA
)";

}  // namespace sor::test
