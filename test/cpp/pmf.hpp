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
  pmf(U value) noexcept
    : _value(static_cast<T>(value))
  {
  }
  constexpr T value() const noexcept { return _value; }
  constexpr pmf& operator+=(pmf<T> other) noexcept
  {
    _value += other._value;
    return *this;
  }

  friend constexpr pmf<T> operator*(pmf<T> a, pmf<T> b) noexcept
  {
    return pmf<T>(a._value * b._value);
  }

  friend constexpr pmf<T> operator/(pmf<T> a, pmf<T> b) noexcept
  {
    return pmf<T>(a._value / b._value);
  }

  friend constexpr pmf<T> operator+(pmf<T> a, pmf<T> b) noexcept
  {
    return pmf<T>(a._value + b._value);
  }

  friend constexpr pmf<T> operator-(pmf<T> a) noexcept
  {
    return pmf<T>(-a._value);
  }
};

namespace std {
template<typename T>
constexpr pmf<T>
abs(const pmf<T>& a) noexcept
{
  return pmf<T>(std::abs(a.value()));
}
}
