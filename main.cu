#include <iostream>
#include <chrono>
#include "method_1.cu"
using namespace std;

string DATA_PATH = "HOSE_Field_2007_2023_lastest.xlsx";
string INTEREST = "1.06";
string VALUEARG_THRESHOLD = "500000000.0";


template <typename T>
void print_array(T *array, int length, int step){
    for (int i=0; i<length; i+=step) cout << round(10000.0*array[i])/10000.0 << " ";
    cout << endl;
}


int main(){
    string command = "python dataToBinary.py --dataPath=" + DATA_PATH
                   + " --interest="                        + INTEREST
                   + " --valueargThreshold="               + VALUEARG_THRESHOLD;
    system(command.c_str());

    // Read bin files
    int *INDEX, *SYMBOL, *BOOL_ARG, index_length, rows, cols;
    double *PROFIT, *OPERAND;
    string index_path = "InputData/INDEX.bin";
    string profit_path = "InputData/PROFIT.bin";
    string symbol_path = "InputData/SYMBOL.bin";
    string bool_arg_path = "InputData/BOOL_ARG.bin";
    string operand_path = "InputData/OPERAND.bin";

    try {
        read_binary_file_1d(INDEX, index_length, index_path);
        read_binary_file_1d(PROFIT, rows, profit_path);
        read_binary_file_1d(SYMBOL, rows, symbol_path);
        read_binary_file_1d(BOOL_ARG, rows, bool_arg_path);
        read_binary_file_2d(OPERAND, cols, rows, operand_path);
    } catch (runtime_error &e){
        cout << e.what();
        return 1;
    }

    // Print arrays
    cout << index_length << " " << rows << " " << cols << endl;
    print_array(INDEX, index_length, 1);
    print_array(PROFIT, rows, 500);
    print_array(SYMBOL, rows, 500);
    print_array(BOOL_ARG, rows, 500);
    print_array(OPERAND+rows, rows, 500);
    print_array(OPERAND+rows*(cols-2), rows, 500);

    //
    Generator vis(INDEX, SYMBOL, BOOL_ARG, index_length, rows, cols, PROFIT, OPERAND);
    
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    vis.test_run(4);
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << duration.count() << endl;

    // Deallocate
    delete[] INDEX;
    delete[] SYMBOL;
    delete[] BOOL_ARG;
    delete[] PROFIT;
    delete[] OPERAND;
}