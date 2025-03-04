#include "Ndarray.h"
#include <numeric>
#include <algorithm>

using namespace std;

Ndarray::Ndarray(const vector<int>& shape, double fill_value) 
    : m_shape(shape), m_size(1) {
    for (int dim : m_shape) m_size *= dim;
    
    m_data.resize(m_size, fill_value);

    m_strides.resize(m_shape.size());

    int stride = 1;                                 //satria 1 fona ny pas voalohany...
                                                    //...(le mikisaka eo akaiky eo am le collone manarakany)

    for (int i = m_shape.size() - 1; i >= 0; --i) { //formule tsy voazava tsara revo miotran 2D fa de mety hicalculena azy
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}



Ndarray::Ndarray(const vector<double>& data, const vector<int>& shape) 
: m_data(data), m_shape(shape), m_size(data.size()) {
    if (m_size != accumulate(shape.begin(), shape.end(), 1, multiplies<int>()))
        throw invalid_argument("Shape doesn't match data size");
    
    m_strides.resize(m_shape.size());

    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; --i){
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}



Ndarray::Ndarray(initializer_list<double> data, const vector<int>& shape) 
: m_data(data), m_shape(shape), m_size(data.size()) {
    if (m_size != accumulate(shape.begin(), shape.end(), 1, multiplies<int>()))
        throw invalid_argument("Shape doesn't match data size");
    
    m_strides.resize(m_shape.size());

    int stride = 1;
    for (int i = m_shape.size() - 1; i >= 0; --i){
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

Ndarray::~Ndarray()
{

}

//------------------><------------------
Ndarray Ndarray::zeros(const vector<int>& shape) {
    return Ndarray(shape, 0.0);
}

Ndarray Ndarray::ones(const vector<int>& shape) {
    return Ndarray(shape, 1.0);
}

Ndarray Ndarray::arange(double start, double stop, double step) {
    vector<double> data;
    for (double val = start; val < stop; val += step)
        data.push_back(val);
    return Ndarray(data, {(int)data.size()});
}

//>---accès---<
double& Ndarray::at(const vector<int>& indices) {
    int offset = compute_offset(indices);
    return m_data[offset];
}

double Ndarray::at(const vector<int>& indices) const{
    int offset = compute_offset(indices);
    return m_data[offset];
}

// >--Opérations--<
Ndarray Ndarray::operator+(const Ndarray& other) const{
    check_m_shapecompatibility(other);
    vector<double> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] + other.m_data[i];

    return Ndarray(result_data, m_shape);
}

Ndarray Ndarray::operator-(const Ndarray& other) const{
    check_m_shapecompatibility(other);
    vector<double> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] - other.m_data[i];
    
    return Ndarray(result_data, m_shape);
}

Ndarray Ndarray::operator*(const Ndarray& other) const{
    // multiplication matricielle
}

Ndarray Ndarray::operator+(double scalar) const{
    vector<double> result_data(m_size);
    for (int i = 0; i < m_size; ++i)
        result_data[i] = m_data[i] + scalar;

    return Ndarray(result_data, m_shape);
}

// Manipulation
Ndarray Ndarray::reshape(const vector<int>& new_shape) const{
    int new_size = accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>());
    if (new_size != m_size)
        throw invalid_argument("New shape does not match total size");
    
    return Ndarray(m_data, new_shape);
}

Ndarray Ndarray::transpose() const{
    if (m_shape.size() != 2)
        throw runtime_error("Transpose is only supported for 2D arrays");
    
    vector<int> new_shape = {m_shape[1], m_shape[0]};
    vector<double> new_data(m_size);
    for (int i = 0; i < m_shape[0]; ++i) 
        for (int j = 0; j < m_shape[1]; ++j) 
            new_data[j * m_shape[0] + i] = m_data[i * m_shape[1] + j];
        
    
    return Ndarray(new_data, new_shape);
}

// Fonctions mathématiques
double Ndarray::sum() const{
    return accumulate(m_data.begin(), m_data.end(), 0.0);
}

double Ndarray::max() const{
    return *max_element(m_data.begin(), m_data.end());
}

double Ndarray::min() const{
    return *min_element(m_data.begin(), m_data.end());
}

// Utilitaires internes
int Ndarray::compute_offset(const vector<int>& indices) const {
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

void Ndarray::check_m_shapecompatibility(const Ndarray& other) const {
    if (m_shape != other.m_shape)
        throw invalid_argument("Shapes are not compatible");
}