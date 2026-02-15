#include <gtest/gtest.h>

#include "poisson.hpp"

TEST(Kernel, Integral)
{
  // TODO: add assembly tests
  EXPECT_EQ(7 * 6, 42);
  integral_3af066e0aa4a1ce87756d2984331c55a2d3d2f62 integral;

  using scalar_t = double;
  using geo_t = double;

  std::array<scalar_t, 9> A{ 0 };
  std::array<scalar_t, 0> w;
  std::array<scalar_t, 4> c{ 1, 2, 3, 4 };
  std::array<geo_t, 9> coords{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };
  std::array<geo_t, 0> empty;

  integral.tabulate_tensor<double, double>(
    A.data(), w.data(), c.data(), coords.data(), 0, 0);

  std::array<scalar_t, 9> A_expected{
    5, -2.5, -2.5, -2.5, 2.5, 0, -2.5, 0, 2.5
  };

  for (std::size_t i = 0; i < A.size(); ++i)
    EXPECT_DOUBLE_EQ(A[i], A_expected[i]);
}
