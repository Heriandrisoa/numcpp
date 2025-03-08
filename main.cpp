#include "Ndarray.h"
#include <iostream>

using namespace std;

void print_array(const Ndarray& arr, const string& name) {
    cout << name << " (shape: ";
    auto shape = arr.get_shape();
    for (int i = 0; i < shape.size(); ++i) {
        cout << shape[i];
        if (i < shape.size() - 1)
            cout << ", ";
    }
    cout << "):"<<endl;
    
    if (shape.size() == 1) {
        for (int i = 0; i < shape[0]; ++i) {
            cout << arr.at({i}) << " ";
        }
        cout << "\n";
    } else if (shape.size() == 2) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                cout << arr.at({i, j}) << " ";
            }
            cout << "\n";
        }
    }
    cout << "\n";
}

int main() {
    try {
        Ndarray tableau({1,2,3,4});
    } catch (const exception& e) {
        cerr << "Erreur : " << e.what() << endl;
        return 1;
    }

    return 0;
}