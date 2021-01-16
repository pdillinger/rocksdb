//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "util/ribbon_config.h"

namespace ROCKSDB_NAMESPACE {

namespace ribbon {

namespace detail {

// Based on data from FindOccupancyForSuccessRate in ribbon_test

template <>
const double BandingConfigHelperData1<128U>::kFactorPerPow2 = 0.0038;
template <>
const double BandingConfigHelperData1<64U>::kFactorPerPow2 = 0.0083;

template <>
const double BandingConfigHelperData2<kOneIn2, 128U>::kBaseFactor = 0.9570;
template <>
const double BandingConfigHelperData2<kOneIn20, 128U>::kBaseFactor = 0.9712;
template <>
const double BandingConfigHelperData2<kOneIn1000, 128U>::kBaseFactor = 0.9916;
template <>
const double BandingConfigHelperData2<kOneIn2, 64U>::kBaseFactor = 0.9219;
template <>
const double BandingConfigHelperData2<kOneIn20, 64U>::kBaseFactor = 0.9529;
template <>
const double BandingConfigHelperData2<kOneIn1000, 64U>::kBaseFactor = 0.9993;

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn2, 128U, false>::kKnownByPow2 = {
        0,       0,       0,       0,       0,
        0,       0,       0,  // unsupported
        252.984, 506.109, 1013.71, 2029.47, 4060.43,
        8115.63, 16202.2, 32305.1, 64383.5, 128274,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn2, 128U, /*smash*/ true>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,       0,  // unsupported
        126.274, 254.279, 510.27,  1022.24, 2046.02, 4091.99, 8154.98,
        16244.3, 32349.7, 64426.6, 128307,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn2, 64U, false>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,       0,  // unsupported
        124.94,  249.968, 501.234, 1004.06, 2006.15, 3997.89, 7946.99,
        15778.4, 31306.9, 62115.3, 123284,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn2, 64U, /*smash*/ true>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,  // unsupported
        62.2683, 126.259, 254.268, 509.975, 1019.98, 2026.16,
        4019.75, 7969.8,  15798.2, 31330.3, 62134.2, 123255,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn20, 128U, false>::kKnownByPow2 = {
        0,       0,       0,       0,       0,
        0,       0,       0,  // unsupported
        248.851, 499.532, 1001.26, 2003.97, 4005.59,
        8000.39, 15966.6, 31828.1, 63447.3, 126506,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn20, 128U, /*smash*/ true>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,      0,  // unsupported
        122.637, 250.651, 506.625, 1018.54, 2036.43, 4041.6, 8039.25,
        16005,   31869.6, 63492.8, 126537,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn20, 64U, false>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,       0,  // unsupported
        120.659, 243.346, 488.168, 976.373, 1948.86, 3875.85, 7704.97,
        15312.4, 30395.1, 60321.8, 119813,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn20, 64U, /*smash*/ true>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,  // unsupported
        58.6016, 122.619, 250.641, 503.595, 994.165, 1967.36,
        3898.17, 7727.21, 15331.5, 30405.8, 60376.2, 119836,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn1000, 128U, false>::kKnownByPow2 = {
        0,       0,       0,       0,       0,
        0,       0,       0,  // unsupported
        242.61,  491.887, 983.603, 1968.21, 3926.98,
        7833.99, 15629,   31199.9, 62307.8, 123870,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn1000, 128U, /*smash*/ true>::kKnownByPow2 = {
        0,      0,       0,       0,       0,      0,       0,  // unsupported
        117.19, 245.105, 500.748, 1010.67, 1993.4, 3950.01, 7863.31,
        15652,  31262.1, 62462.8, 124095,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn1000, 64U, false>::kKnownByPow2 = {
        0,     0,     0,       0,       0,    0,      0,  // unsupported
        114,   234.8, 471.498, 940.165, 1874, 3721.5, 7387.5,
        14592, 29160, 57745,   115082,
};

template <>
const std::array<double, 18>
    BandingConfigHelperData3<kOneIn1000, 64U, /*smash*/ true>::kKnownByPow2 = {
        0,       0,       0,       0,       0,       0,  // unsupported
        53.0434, 117,     245.312, 483.571, 950.251, 1878,
        3736.34, 7387.97, 14618,   29142.9, 57838.8, 114932,
};

}  // namespace detail

}  // namespace ribbon

}  // namespace ROCKSDB_NAMESPACE
