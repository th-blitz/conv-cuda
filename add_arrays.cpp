#include <cstdlib>
#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

void print_array(float* array, int size, const char* name) {
    cout << name << " : " << "size : " << size << endl;
    for (int i = 0; i < size; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

void add_arrays(float* array_a, float* array_b, float* array_c, int size) {
    for (int i = 0; i < size; i++) {
        array_c[i] = array_a[i] + array_b[i];
    }
}


int main(int argc, char* argv[]) {
    
    std::cout << "Number of arguments: " << argc << std::endl;

    std::cout << "Arguments:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    int k = stoi(argv[1]);
    int size = k * 1000000;

    float* array_a = (float*)malloc(size * sizeof(float));
    float* array_b = (float*)malloc(size * sizeof(float));
    float* array_c = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        array_a[i] = i;
        array_b[i] = size - 1 - i;
        array_c[i] = 0;
    }

    auto start = high_resolution_clock::now();
   
    add_arrays(array_a, array_b, array_c, size);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Time taken for K = " << k << " million elements: " << duration.count() << " milliseconds" << endl;

//    print_array(array_c, size, "array_c");

    free(array_a);
    free(array_b);
    free(array_c);


    return 0;
}
