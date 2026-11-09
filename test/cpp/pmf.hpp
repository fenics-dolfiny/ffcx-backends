#pragma once
#include <cmath>

template<typename T>
class pmf
{
private:
  T _value;

public:
  pmf()
    : _value{}
  {
  }
  template<typename U>
  explicit(false) pmf(U value) noexcept
    : _value(static_cast<T>(value))
  {
  }
  [[nodiscard]] constexpr auto value() const noexcept -> T { return _value; }
  constexpr auto operator+=(pmf<T> other) noexcept -> pmf&
  {
    _value += other._value;
    return *this;
  }

  friend constexpr auto operator*(pmf<T> a, pmf<T> b) noexcept -> pmf<T>
  {
    return pmf<T>(a._value * b._value);
  }

  friend constexpr auto operator/(pmf<T> a, pmf<T> b) noexcept -> pmf<T>
  {
    return pmf<T>(a._value / b._value);
  }

  friend constexpr auto operator+(pmf<T> a, pmf<T> b) noexcept -> pmf<T>
  {
    return pmf<T>(a._value + b._value);
  }

  friend constexpr auto operator-(pmf<T> a) noexcept -> pmf<T>
  {
    return pmf<T>(-a._value);
  }
};

namespace std {
template<typename T>
constexpr auto
abs(const pmf<T>& a) noexcept -> pmf<T>
{
  return pmf<T>(std::abs(a.value()));
}
} // namespace std
