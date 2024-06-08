#include <iostream>
#include <fstream>
using namespace std;


void write_formula(uint8_t *formula, int num_opr_per_fml, ofstream &outFile){
    for (int i=0; i<num_opr_per_fml; i++){
        switch (formula[2*i]){
            case 0:
                outFile << "+";
                break;
            case 1:
                outFile << "-";
                break;
            case 2:
                outFile << "*";
                break;
            case 3:
                outFile << "/";
                break;
            default:
                throw runtime_error("Sai format cong thuc");
        }
        outFile << static_cast<int>(formula[2*i+1]);
    }
    outFile << endl;
}


template <typename T>
void read_binary_file_1d(T *&array, int &length, string path){
    ifstream file(path, ios::binary);
    if (file.is_open()){
        file.read(reinterpret_cast<char*>(&length), 4);
        array = new T[length];
        size_t t = sizeof(T);
        for (int i=0; i<length; i++){
            file.read(reinterpret_cast<char*>(&array[i]), t);
            if (file.gcount() < t) throw runtime_error("Do dai file khong du");
        }
        file.close();
    } else throw runtime_error("Khong mo duoc file");
}


template <typename T>
void read_binary_file_2d(T *&array, int &rows, int &cols, string path){
    ifstream file(path, ios::binary);
    if (file.is_open()){
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        array = new T[rows*cols];
        size_t t = sizeof(T);
        for (int i=0; i<rows*cols; i++){
            file.read(reinterpret_cast<char*>(&array[i]), t);
            if (file.gcount() < t) throw runtime_error("Do dai file khong du");
        }
        file.close();
    } else throw runtime_error("Khong mo duoc file");
}
