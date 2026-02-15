#include <gtest/gtest.h>

#include "poisson.hpp"

template<typename T>
struct real_type
{
  using type = T;
};

template<typename T>
struct real_type<std::complex<T>>
{
  using type = T;
};

template<typename T>
class Kernel : public ::testing::Test
{};

using ScalarTypes =
  ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Kernel, ScalarTypes);

TYPED_TEST(Kernel, Integral)
{
  // TODO: add assembly tests
  EXPECT_EQ(7 * 6, 42);
  integral_3af066e0aa4a1ce87756d2984331c55a2d3d2f62 integral;

  using scalar_t = TypeParam;
  using geo_t = real_type<scalar_t>::type; // TODO: real of scalar_t

  std::array<scalar_t, 9> A{ 0 };
  std::array<scalar_t, 0> w;
  std::array<scalar_t, 4> c{ 1, 2, 3, 4 };
  std::array<geo_t, 9> coords{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };
  std::array<geo_t, 0> empty;

  integral.tabulate_tensor<scalar_t, geo_t>(
    A.data(), w.data(), c.data(), coords.data(), 0, 0);

  std::array<scalar_t, 9> A_expected{
    5, -2.5, -2.5, -2.5, 2.5, 0, -2.5, 0, 2.5
  };

  for (std::size_t i = 0; i < A.size(); ++i) {
    if constexpr (std::is_same_v<std::complex<geo_t>, scalar_t>) {
      EXPECT_DOUBLE_EQ(std::real(A[i]), std::real(A_expected[i]));
      EXPECT_DOUBLE_EQ(std::imag(A[i]), std::imag(A_expected[i]));
    } else {
      EXPECT_DOUBLE_EQ(A[i], A_expected[i]);
    }
  }
}
