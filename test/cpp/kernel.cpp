#include <gtest/gtest.h>

#include "poisson.hpp"

template<typename T>
struct real
{
  using type = T;
};

template<typename T>
struct real<std::complex<T>>
{
  using type = T;
};

template<typename T>
using real_t = real<T>::type;

// Helper function for type-appropriate equality checks
template<typename T>
void expect_eq(T a, T b)
{
  if constexpr (std::is_same_v<float, T>) {
    EXPECT_FLOAT_EQ(a, b);
  } else {
    EXPECT_DOUBLE_EQ(a, b);
  }
}

template<typename T>
void expect_eq(std::complex<T> a, std::complex<T> b)
{
  expect_eq(std::real(a), std::real(b));
  expect_eq(std::imag(a), std::imag(b));
}

template<typename T>
class Kernel : public ::testing::Test
{};

using ScalarTypes =
  ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Kernel, ScalarTypes);

TYPED_TEST(Kernel, Integral)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  form_poisson_a::triangle_integral integral_a;
  form_poisson_L::triangle_integral integral_L;

  // Bilinear form test data
  std::array<scalar_t, 9> A{ 0 };
  const std::array<scalar_t, 0> w_a;
  const std::array<scalar_t, 4> c{ 1, 2, 3, 4 };
  const std::array<geo_t, 9> coords{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };

  integral_a.tabulate_tensor<scalar_t, geo_t>(
    A.data(), w_a.data(), c.data(), coords.data(), 0, 0);

  const std::array<scalar_t, 9> A_expected{ 5, -2.5, -2.5, -2.5, 2.5,
                                            0, -2.5, 0,    2.5 };

  for (std::size_t i = 0; i < A.size(); ++i) {
    expect_eq(A[i], A_expected[i]);
  }

  // Linear form test data
  std::array<scalar_t, 3> b{ 0 };
  const std::array<scalar_t, 3> w_L{ 1, 1, 1 };  // Coefficient f = 1 at all nodes
  const std::array<scalar_t, 0> c_L;  // No constants for form L

  integral_L.tabulate_tensor<scalar_t, geo_t>(
    b.data(), w_L.data(), c_L.data(), coords.data(), 0, 0);

  // For f = 1 (constant), the expected result is the integral of each basis function
  // over the reference triangle with area 1/2. Each basis function integrates to 1/6.
  const std::array<scalar_t, 3> b_expected{ 1.0/6.0, 1.0/6.0, 1.0/6.0 };

  for (std::size_t i = 0; i < b.size(); ++i) {
    expect_eq(b[i], b_expected[i]);
  }
}
