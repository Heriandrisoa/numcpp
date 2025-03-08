#include "Ndarray.h"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>

template <typename T>
Ndarray<T>::Ndarray(const vector<int>& shape, T fill_value) 
    : m_shape(shape), m_size(1) {
    for (int dim : m_shape) m_size *= dim;
    m_data.resize(m_size, fill_value);
    
    m_strides.resize(m_shape.size());
    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

template <typename T>
Ndarray<T>::Ndarray(const vector<T>& data, const vector<int>& shape) 
    : m_data(data), m_shape(shape), m_size(data.size()) {
    if (m_size != accumulate(shape.begin(), shape.end(), 1, multiplies<int>()))
        throw invalid_argument("Shape doesn't match data size");

    m_strides.resize(m_shape.size());
    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

template <typename T>
Ndarray<T>::Ndarray(initializer_list<T> data, const vector<int>& shape) 
    : m_data(data), m_shape(shape), m_size(data.size()) {
    if (m_size != accumulate(shape.begin(), shape.end(), 1, multiplies<int>()))
        throw invalid_argument("Shape doesn't match data size");

    m_strides.resize(m_shape.size());
    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

template <typename T>
template<typename... t> 
Ndarray<T>::Ndarray(t... tp) : m_size(1) {
    // Create a list of vectors from the variadic arguments
    initializer_list<vector<T>> type = {tp...};
    
    int norm_size = type.begin()->size();  // Get the size of the first vector
    for (const auto& i : type) {
        if (i.size() != norm_size) {
            throw invalid_argument("All rows must be of the same length");
        }
        m_data.insert(m_data.end(), i.begin(), i.end()); // Merge all vectors into m_data
    }

    m_shape = { static_cast<int>(type.size()), norm_size };

    for (int dim : m_shape)
        m_size *= dim;

    m_strides.resize(m_shape.size());
    int strides = 1;
    for (int i = m_shape.size() - 1; i > 0; i--) {
        m_strides[i] = strides;
        strides *= m_shape[i];
    }
}

// Générateurs
template <typename T>
Ndarray<T> Ndarray<T>::zeros(const vector<int>& shape) {
    return Ndarray<T>(shape, 0);
}

template <typename T>
Ndarray<T> Ndarray<T>::ones(const vector<int>& shape) {
    return Ndarray<T>(shape, 1);
}

template <typename T>
Ndarray<T> Ndarray<T>::arange(T start, T stop, T step) {
    vector<T> data;
    for (T val = start; val < stop; val += step)
        data.push_back(val);
    return Ndarray<T>(data, {(int)data.size()});
}

// Accès
template <typename T>
T& Ndarray<T>::at(const vector<int>& indices) {
    int offset = compute_offset(indices);
    return m_data[offset];
}

template <typename T>
T Ndarray<T>::at(const vector<int>& indices) const {
    int offset = compute_offset(indices);
    return m_data[offset];
}

// Opérations
template <typename T>
Ndarray<T> Ndarray<T>::operator+(const Ndarray& other) const {
    check_shape_compatibility(other);
    vector<T> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] + other.m_data[i];

    return Ndarray<T>(result_data, m_shape);
}

template <typename T>
Ndarray<T> Ndarray<T>::operator-(const Ndarray& other) const {
    check_shape_compatibility(other);
    vector<T> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] - other.m_data[i];

    return Ndarray<T>(result_data, m_shape);
}

template <typename T>
Ndarray<T> Ndarray<T>::operator+(T scalar) const {
    vector<T> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] + scalar;

    return Ndarray<T>(result_data, m_shape);
}

// Manipulation
template <typename T>
Ndarray<T> Ndarray<T>::reshape(const vector<int>& new_shape) const {
    int new_size = accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>());
    if (new_size != m_size)
        throw invalid_argument("New shape does not match total size");

    return Ndarray<T>(m_data, new_shape);
}

template <typename T>
Ndarray<T> Ndarray<T>::transpose() const {
    if (m_shape.size() != 2)
        throw runtime_error("Transpose is only supported for 2D arrays");

    vector<int> new_shape = {m_shape[1], m_shape[0]};
    vector<T> new_data(m_size);
    for (int i = 0; i < m_shape[0]; ++i) 
        for (int j = 0; j < m_shape[1]; ++j) 
            new_data[j * m_shape[0] + i] = m_data[i * m_shape[1] + j];

    return Ndarray<T>(new_data, new_shape);
}

// Fonctions mathématiques
template <typename T>
T Ndarray<T>::sum() const {
    return accumulate(m_data.begin(), m_data.end(), (T)0);
}

template <typename T>
T Ndarray<T>::max() const {
    return *max_element(m_data.begin(), m_data.end());
}

template <typename T>
T Ndarray<T>::min() const {
    return *min_element(m_data.begin(), m_data.end());
}

// Utilitaires internes
template <typename T>
int Ndarray<T>::compute_offset(const vector<int>& indices) const {
    if (indices.size() != m_shape.size())
        throw out_of_range("Invalid number of indices");

    int offset = 0;
    for (int i = 0; i < indices.size(); ++i) {
        if (indices[i] >= m_shape[i])
            throw out_of_range("Index out of bounds");
        offset += indices[i] * m_strides[i];
    }
    return offset;
}

template <typename T>
void Ndarray<T>::check_shape_compatibility(const Ndarray& other) const {
    if (m_shape != other.m_shape)
        throw invalid_argument("Shapes are not compatible");
}

