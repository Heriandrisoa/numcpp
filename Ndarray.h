#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

template <typename T>
class Ndarray {
private:
    vector<T> m_data;
    vector<int> m_shape;
    vector<int> m_strides;
    int m_size;

    int compute_offset(const vector<int>& indices) const;
    void check_shape_compatibility(const Ndarray& other) const;

public:
    Ndarray(const vector<int>& shape, T fill_value);
    Ndarray(const vector<T>& data, const vector<int>& shape);
    Ndarray(initializer_list<T> data, const vector<int>& shape);

    ~Ndarray() = default;

    // Générateurs
    static Ndarray<T> zeros(const vector<int>& shape);
    static Ndarray<T> ones(const vector<int>& shape);
    static Ndarray<T> arange(T start, T stop, T step);

    // Opérations
    Ndarray<T> operator+(const Ndarray& other) const;
    Ndarray<T> operator-(const Ndarray& other) const;
    Ndarray<T> operator*(const Ndarray& other) const; // À implémenter
    Ndarray<T> operator+(T scalar) const;

    // Accès
    T& at(const vector<int>& indices);
    T at(const vector<int>& indices) const;

    // Manipulation
    Ndarray<T> reshape(const vector<int>& new_shape) const;
    Ndarray<T> transpose() const;

    // Math
    T sum() const;
    T max() const;
    T min() const;
};