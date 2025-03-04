#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>

class Ndarray {
public:
    Ndarray(const std::vector<int>&dshape, double fill_value = 0);
    Ndarray(const std::vector<double>& data, const std::vector<int>& shape);
    Ndarray(const std::initializer_list<double> list_data , const std::vector<int>& shape);
    ~Ndarray(); 

    static Ndarray zeros(const std::vector<int>& shape);
    static Ndarray ones(const std::vector<int>& shape);
    static Ndarray arange(double start, double stop, double step);

    double& at(const std::vector<int>& indices);        //raha modifiena
    double at(const std::vector<int>& indices) const;   //raha récupérena

    Ndarray operator+(const Ndarray& other) const;
    Ndarray operator-(const Ndarray& other) const;
    Ndarray operator*(const Ndarray& other) const;
    Ndarray operator+(double scalar) const;

    Ndarray reshape(const std::vector<int>& new_shape) const;
    Ndarray transpose() const;
    double sum() const;
    double max() const;
    double min() const;

    // Getters
    std::vector<int> get_shape() const { return m_shape; }
    int size() const { return m_size; }

private:
    std::vector<double> m_data;      // Données stockées
    std::vector<int> m_shape;     // dimension du tableau
    int m_size;                   // Nombre total d'éléments
    std::vector<int> m_strides;   // map de dimension

    //utilitaires
    int compute_offset(const std::vector<int>& indices) const;
    void check_m_shapecompatibility(const Ndarray& other) const;
};

#endif 