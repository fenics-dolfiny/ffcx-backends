#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

#include "pmf.hpp"

#include "interpolation.hpp"
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
struct real<pmf<T>>
{
  using type = pmf<T>;
};

template<typename T>
using real_t = real<T>::type;

template<typename T>
class Kernel : public ::testing::Test
{};

void
EXPECT_SCALAR_EQ(float a, float b)
{
  EXPECT_FLOAT_EQ(a, b);
}

void
EXPECT_SCALAR_EQ(double a, double b)
{
  EXPECT_DOUBLE_EQ(a, b);
}

template<typename T>
void
EXPECT_SCALAR_EQ(const std::complex<T>& a, const std::complex<T>& b)
{
  EXPECT_SCALAR_EQ(std::real(a), std::real(b));
  EXPECT_SCALAR_EQ(std::imag(a), std::imag(b));
}

template<typename T>
void
EXPECT_SCALAR_EQ(const pmf<T>& a, const pmf<T>& b)
{
  EXPECT_SCALAR_EQ(a.value(), b.value());
}

// Equal up to a few rounding errors, relative to the expected value b.
template<std::floating_point T>
void
EXPECT_SCALAR_NEAR(T a, T b)
{
  const T tol =
    16 * std::numeric_limits<T>::epsilon() * std::max(T{ 1 }, std::abs(b));
  EXPECT_NEAR(a, b, tol);
}

template<typename T>
void
EXPECT_SCALAR_NEAR(const std::complex<T>& a, const std::complex<T>& b)
{
  EXPECT_SCALAR_NEAR(std::real(a), std::real(b));
  EXPECT_SCALAR_NEAR(std::imag(a), std::imag(b));
}

template<typename T>
void
EXPECT_SCALAR_NEAR(const pmf<T>& a, const pmf<T>& b)
{
  EXPECT_SCALAR_NEAR(a.value(), b.value());
}

using ScalarTypes = ::testing::Types<float,
                                     double,
                                     std::complex<float>,
                                     std::complex<double>,
                                     pmf<float>,
                                     pmf<double>,
                                     pmf<std::complex<float>>,
                                     pmf<std::complex<double>>>;
TYPED_TEST_SUITE(Kernel, ScalarTypes);

TYPED_TEST(Kernel, Tensor)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const typename form_poisson_a<scalar_t, geo_t>::integral_triangle_all
    integral_a;

  std::array<scalar_t, 9> A{ 0 };
  const std::array<scalar_t, 0> w_a{};
  const std::array<scalar_t, 4> c{ 1, 2, 3, 4 };
  const std::array<geo_t, 9> coords{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };

  integral_a.tabulate_tensor(
    A.data(), w_a.data(), c.data(), coords.data(), nullptr, nullptr);

  const std::array<scalar_t, 9> A_expected{
    5, -2.5, -2.5, -2.5, 2.5, 0, -2.5, 0, 2.5,
  };

  for (std::size_t i = 0; i < A.size(); ++i) {
    EXPECT_SCALAR_EQ(A[i], A_expected[i]);
  }
}

TYPED_TEST(Kernel, Vector)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const typename form_poisson_L<scalar_t, geo_t>::integral_triangle_all
    integral_L;

  std::array<scalar_t, 3> b{ 0 };
  const std::array<scalar_t, 3> w_L{ 1, 2, 3 };
  const std::array<scalar_t, 0> c_L{};
  const std::array<geo_t, 9> coords{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };

  integral_L.tabulate_tensor(
    b.data(), w_L.data(), c_L.data(), coords.data(), nullptr, nullptr);

  const std::array<scalar_t, 3> b_expected{
    7.0 / 24.0,
    8.0 / 24.0,
    9.0 / 24.0,
  };

  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_SCALAR_EQ(b[i], b_expected[i]);
  }
}

TYPED_TEST(Kernel, Expression)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const expression_poisson_0<scalar_t, geo_t> expr;

  std::array<scalar_t, 18> e{ 0 };
  const std::array<scalar_t, 22> w{
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
  };
  const std::array<scalar_t, 4> c{
    1,
    2,
    3,
    4,
  };
  const std::array<geo_t, 9> coords{
    0, 0, 0, 1, 0, 0, 0, 1, 0,
  };
  expr.tabulate_tensor(
    e.data(), w.data(), c.data(), coords.data(), nullptr, nullptr, nullptr);

  std::array<scalar_t, 18> e_expected{
    5, 7, 8, 11, 15, 18, 14, 16, 17, 32, 36, 39, 23, 25, 26, 53, 57, 60,
  };

  for (std::size_t i = 0; i < e.size(); ++i) {
    EXPECT_SCALAR_EQ(e[i], e_expected[i]);
  }
}

// The interpolation tests share a coefficient, in P2, and a cell of area 3,
// see interpolation.py.

TYPED_TEST(Kernel, InterpolationProxyCoefficient)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const typename form_interpolation_proxy_coefficient<scalar_t, geo_t>::
    integral_triangle_all integral;

  std::array<scalar_t, 1> A{ 0 };
  const std::array<scalar_t, 0> c{};
  const std::array<scalar_t, 6> w{ 0.5, -1.0, 2.0, 0.3, 0.7, -0.2 };
  const std::array<geo_t, 9> coords{ 0, 0, 0, 2, 0, 0, 0, 3, 0 };

  integral.tabulate_tensor(
    A.data(), w.data(), c.data(), coords.data(), nullptr, nullptr);

  // area / 3 * (w_0^2 + w_1^2 + w_2^2)
  EXPECT_SCALAR_NEAR(A[0], scalar_t(5.25));
}

TYPED_TEST(Kernel, InterpolationRuntimeTable)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const typename form_interpolation_runtime_table<scalar_t,
                                                  geo_t>::integral_triangle_all
    integral;

  std::array<scalar_t, 6> A{ 0 };
  const std::array<scalar_t, 0> c{};
  const std::array<scalar_t, 6> w{ 0.5, -1.0, 2.0, 0.3, 0.7, -0.2 };
  const std::array<geo_t, 9> coords{ 0, 0, 0, 2, 0, 0, 0, 3, 0 };

  integral.tabulate_tensor(
    A.data(), w.data(), c.data(), coords.data(), nullptr, nullptr);

  // area / 3 * w_i at the vertices, the edge dofs vanish there
  const std::array<scalar_t, 6> A_expected{ 0.5, -1.0, 2.0, 0, 0, 0 };

  for (std::size_t i = 0; i < A.size(); ++i) {
    EXPECT_SCALAR_NEAR(A[i], A_expected[i]);
  }
}

TYPED_TEST(Kernel, InterpolationNestedExpression)
{
  using scalar_t = TypeParam;
  using geo_t = real_t<scalar_t>;

  const expression_interpolation_nested<scalar_t, geo_t> expr;

  std::array<scalar_t, 3> e{ 0 };
  const std::array<scalar_t, 1> c{ 2.5 };
  const std::array<scalar_t, 6> w{ 0.5, -1.0, 2.0, 0.3, 0.7, -0.2 };
  const std::array<geo_t, 9> coords{ 0, 0, 0, 2, 0, 0, 0, 3, 0 };

  expr.tabulate_tensor(
    e.data(), w.data(), c.data(), coords.data(), nullptr, nullptr, nullptr);

  // The P1 interpolant of c * w_i^3, at the points
  const std::array<scalar_t, 3> e_expected{ 2.0, 2.5625, 9.453125 };

  for (std::size_t i = 0; i < e.size(); ++i) {
    EXPECT_SCALAR_NEAR(e[i], e_expected[i]);
  }
}
