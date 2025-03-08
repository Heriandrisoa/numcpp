#include "Ndarray.h"
#include <iostream>

using namespace std;

int main() {
    try {
        Ndarray<vector<int>> tableau({1,2,3},{1,4,5});

//        cout << tableau.shape()[0] <<"   "<<tableau.shape()[1] <<endl;
    } catch (const exception& e) {
        cerr << "Erreur : " << e.what() << endl;
        return 1;
    }

    return 0;
}