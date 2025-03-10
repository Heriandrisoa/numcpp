#include <iostream>
#include "Ndarray.h"
using namespace std;

int main() {
    try {
        // Construction correcte d'un Ndarray 2D
        Ndarray<int, 2> a = Ndarray<int, 2>::zeros();
    } catch (const exception& e) {
        cerr << "Erreur : " << e.what() << endl;
        return 1;
    }

    return 0;
}
