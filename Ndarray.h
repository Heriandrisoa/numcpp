#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

template <typename T, int... dim>
class Ndarray {
private:
    vector<T> m_data;
    vector<int> m_shape;
    vector<int> m_strides;
    int m_size;

    int compute_offset(const vector<int>& indices) const;
    void check_shape_compatibility(const Ndarray& other) const;

public:
    Ndarray(T fill_value);
    Ndarray(const vector<T>& data, const vector<int>& shape);
    Ndarray(initializer_list<T> data, const vector<int>& shape);
    Ndarray(initializer_list<initializer_list<T>> data);

    ~Ndarray() = default;

    // Générateurs
    static Ndarray<T,dim...> zeros();
    static Ndarray<T,dim...> ones();
    static Ndarray<T,dim...> arange(T start, T stop, T step);

    // Opérations
    Ndarray<T,dim...> operator+(const Ndarray& other) const;
    Ndarray<T,dim...> operator-(const Ndarray& other) const;
    Ndarray<T,dim...> operator*(const Ndarray& other) const; // À implémenter
    Ndarray<T,dim...> operator+(T scalar) const;

    // Accès
    T& at(const vector<int>& indices);
    T at(const vector<int>& indices) const;


    vector<int> shape() const { return m_shape;}
    // Manipulation
    Ndarray<T,dim...> reshape(const vector<int>& new_shape) const;
    Ndarray<T,dim...> transpose() const;

    // Math
    T sum() const;
    T max() const;
    T min() const;
};