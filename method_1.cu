#include "kernel.cu"
#include "suppFunc.cpp"
#include "workWithFile.cpp"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
using namespace std;

int __STORAGE_SIZE__ = 10000;
int __NUM_CYCLE__ = 10;
double __INTEREST__ = 1.06;


class Generator {
public:
    int *INDEX, *SYMBOL, *BOOL_ARG, index_length, rows, cols;
    double *PROFIT, *OPERAND, *temp_weight_storage;
    int groups, fml_shape, num_per_grp, count_temp_storage = 0;
    int **current;
    uint8_t **temp_formula_storage;
    ofstream outFile;
    ofstream *array_outFile;

    // Nguong 2
    double *d_threshold;
    double *d_result;
    double *d_final;
    double *h_final;

    // Constructor
    Generator(
        int *INDEX,
        int *SYMBOL,
        int *BOOL_ARG,
        int index_length,
        int rows,
        int cols,
        double *PROFIT,
        double *OPERAND
    ) {
        cudaMalloc((void**)&this->INDEX, 4*index_length);
        cudaMemcpy(this->INDEX, INDEX, 4*index_length, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&this->SYMBOL, 4*rows);
        cudaMemcpy(this->SYMBOL, SYMBOL, 4*rows, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&this->BOOL_ARG, 4*rows);
        cudaMemcpy(this->BOOL_ARG, BOOL_ARG, 4*rows, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&this->PROFIT, 8*rows);
        cudaMemcpy(this->PROFIT, PROFIT, 8*rows, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&this->OPERAND, 8*rows*cols);
        cudaMemcpy(this->OPERAND, OPERAND, 8*rows*cols, cudaMemcpyHostToDevice);

        this->index_length = index_length;
        this->rows = rows;
        this->cols = cols;

        //
        outFile.open("Generator_temp_file.txt");
        array_outFile = new ofstream[__NUM_CYCLE__];

        //
        current = new int*[3];
        for (int i=0; i<3; i++) current[i] = new int[1];

        //
        cudaMalloc((void**)&temp_weight_storage, 8*(__STORAGE_SIZE__+cols)*rows);
        temp_formula_storage = new uint8_t*[__STORAGE_SIZE__+cols];

        //
        int num_threshold = 5*(index_length - 2);
        cudaMalloc((void**)&d_threshold, 8*(__STORAGE_SIZE__+cols)*num_threshold);
        cudaMalloc((void**)&d_result, 16*(__STORAGE_SIZE__+cols)*num_threshold*__NUM_CYCLE__);
        cudaMalloc((void**)&d_final, 32*(__STORAGE_SIZE__+cols)*__NUM_CYCLE__);
        cuda_array_assign<<<2*(__STORAGE_SIZE__+cols)*num_threshold*__NUM_CYCLE__/256 + 1, 256>>>(
            d_result, 2*(__STORAGE_SIZE__+cols)*num_threshold*__NUM_CYCLE__, 0
        );
        h_final = new double[4*(__STORAGE_SIZE__+cols)*__NUM_CYCLE__];
    }

    ~Generator(){
        cudaFree(INDEX);
        cudaFree(SYMBOL);
        cudaFree(BOOL_ARG);
        cudaFree(PROFIT);
        cudaFree(OPERAND);

        outFile.close();
        delete[] array_outFile;
        for (int i=0; i<3; i++) delete[] current[i];
        delete[] current;

        cudaFree(temp_weight_storage);
        delete[] temp_formula_storage;

        cudaFree(d_threshold);
        cudaFree(d_result);
        cudaFree(d_final);
        delete[] h_final;
    }

    // Fill formula
    void fill_formula(
        uint8_t *formula,
        int **f_struct,
        int idx,
        double *temp_0,
        int temp_op,
        double *temp_1,
        int mode,
        bool add_sub,
        bool mul_div
    ) {
        // cout << mode << endl;
        if (!mode) /*Sinh dấu cộng trừ*/ {
            int gr_idx = 2147483647, start = 0, i, k;
            bool new_add_sub;
            uint8_t *new_formula = new uint8_t[fml_shape];
            int **new_f_struct = new int*[groups];
            for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

            // Xác định nhóm
            for (i=0; i<groups; i++){
                if (f_struct[i][2]-1 == idx){
                    gr_idx = i;
                    break;
                }
            }

            // Xác định chỉ số bắt đầu
            if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

            // Loop
            for (k=start; k<2; k++){
                memcpy(new_formula, formula, fml_shape);
                for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
                new_formula[idx] = k;
                new_f_struct[gr_idx][0] = k;
                if (k == 1){
                    new_add_sub = true;
                    for (i=gr_idx+1; i<groups; i++){
                        new_formula[new_f_struct[i][2]-1] = 1;
                        new_f_struct[i][0] = 1;
                    }
                } else new_add_sub = false;

                fill_formula(new_formula, new_f_struct, idx+1,
                             temp_0, temp_op, temp_1, 1, new_add_sub, mul_div);
            }

            // Giải phóng bộ nhớ
            delete[] new_formula;
            for (i=0; i<groups; i++) delete[] new_f_struct[i];
            delete[] new_f_struct;
        }
        else if (mode == 2) /*Sinh dấu nhân chia*/ {
            int start = 2, i, j, k;
            bool new_mul_div;
            uint8_t *new_formula = new uint8_t[fml_shape];
            int **new_f_struct = new int*[groups];
            for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

            // Xác định chỉ số bắt đầu
            if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];
            if (!start) start = 2;

            // Loop
            bool *valid_operator = get_valid_operator(f_struct, idx, start);
            for (k=0; k<2; k++){
                if (!valid_operator[k]) continue;
                memcpy(new_formula, formula, fml_shape);
                for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
                new_formula[idx] = k + 2;
                if (k == 1){
                    new_mul_div = true;
                    for (i=idx+2; i<2*new_f_struct[0][1]-1; i+=2) new_formula[i] = 3;
                    for (i=1; i<groups; i++){
                        for (j=0; j<new_f_struct[0][1]-1; j++)
                            new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                    }
                } else {
                    new_mul_div = false;
                    for (i=0; i<groups; i++) new_f_struct[i][3] += 1;
                    if (idx == 2*new_f_struct[0][1]-2){
                        new_mul_div = true;
                        for (i=1; i<groups; i++){
                            for (j=0; j<new_f_struct[0][1]-1; j++)
                                new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                        }
                    }
                }

                fill_formula(new_formula, new_f_struct, idx+1,
                             temp_0, temp_op, temp_1, 1, add_sub, new_mul_div);
            }

            // Giải phóng bộ nhớ
            delete[] valid_operator;
            delete[] new_formula;
            for (i=0; i<groups; i++) delete[] new_f_struct[i];
            delete[] new_f_struct;
        }
        else if (mode == 1){
            int start = 0, i, count = 0;

            // Xác định chỉ số bắt đầu
            if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

            // Xác định các toán hạng hợp lệ và đếm
            bool *valid_operand = get_valid_operand(formula, f_struct, idx, start, cols, groups);
            for (i=0; i<cols; i++){
                if (valid_operand[i]) count++;
            }
            //
            if (count){
                int temp_op_new, new_idx, new_mode, k = 0;
                bool chk = false, temp_0_change;
                double *new_temp_0, *new_temp_1;
                uint8_t **new_formula = new uint8_t*[count];
                int *valid = new int[count];
                int *d_valid;
                int num_block = count*rows/256 + 1;
                cudaMalloc((void**)&d_valid, 4*count);

                for (i=0; i<cols; i++){
                    if (!valid_operand[i]) continue;
                    new_formula[k] = new uint8_t[fml_shape];
                    memcpy(new_formula[k], formula, fml_shape);
                    new_formula[k][idx] = i;
                    valid[k] = i;
                    k++;
                }
                cudaMemcpy(d_valid, valid, 4*count, cudaMemcpyHostToDevice);

                if (formula[idx-1] < 2){
                    temp_op_new = formula[idx-1];
                    if (num_per_grp != 1){
                        // cout << "A1-s";
                        cudaMalloc((void**)&new_temp_1, 8*count*rows);
                        copy_from_operands<<<num_block, 256>>>(new_temp_1, OPERAND, d_valid,
                                                               rows, count);
                        cudaDeviceSynchronize();
                        // cout << "A1-e";
                    }
                } else {
                    temp_op_new = temp_op;
                    cudaMalloc((void**)&new_temp_1, 8*count*rows);
                    if (formula[idx-1] == 2){
                        update_temp_weight<<<num_block, 256>>>(new_temp_1, temp_1, OPERAND,
                                                               d_valid, rows, count, true);
                        cudaDeviceSynchronize();
                    } else {
                        update_temp_weight<<<num_block, 256>>>(new_temp_1, temp_1, OPERAND,
                                                               d_valid, rows, count, false);
                        cudaDeviceSynchronize();
                    }
                }

                for (i=0; i<groups; i++){
                    if (idx+2 == f_struct[i][2]){
                        chk = true;
                        break;
                    }
                }
                if (chk || idx+1 == fml_shape){
                    temp_0_change = true;
                    cudaMalloc((void**)&new_temp_0, 8*count*rows);
                    if (!temp_op_new){
                        if (num_per_grp != 1){
                            // cout << "A1-S" << endl;
                            update_last_weight<<<num_block, 256>>>(new_temp_0, temp_0,
                                                                   new_temp_1, rows,
                                                                   count, true);
                            cudaDeviceSynchronize();
                            // cout << "A1-E" << endl;
                        } else {
                            // cout << "A1-S2" << endl;
                            update_last_weight_through_operands<<<num_block, 256>>>(
                                new_temp_0, temp_0, OPERAND, d_valid, rows, count, true
                            );
                            cudaError_t status = cudaDeviceSynchronize();
                            // cout << "A1-E2" << cudaGetErrorString(status) << endl;
                        }
                    } else {
                        if (num_per_grp != 1){
                            // cout << "A1-S3" << endl;
                            update_last_weight<<<num_block, 256>>>(new_temp_0, temp_0,
                                                                   new_temp_1, rows,
                                                                   count, false);
                            cudaDeviceSynchronize();
                            // cout << "A1-S3" << endl;
                        } else {
                            // cout << "A1-S4" << endl;
                            update_last_weight_through_operands<<<num_block, 256>>>(
                                new_temp_0, temp_0, OPERAND, d_valid, rows, count, false
                            );
                            cudaDeviceSynchronize();
                            // cout << "A1-S4" << endl;
                        }
                    }
                } else temp_0_change = false;

                // cout << "???" << endl;
                if (idx+1 != fml_shape){
                    // cout << "ABC" << endl;
                    if (chk){
                        if (add_sub){
                            new_idx = idx + 2;
                            new_mode = 1;
                        } else {
                            new_idx = idx + 1;
                            new_mode = 0;
                        }
                    } else {
                        if (mul_div){
                            new_idx = idx + 2;
                            new_mode = 1;
                        } else {
                            new_idx = idx + 1;
                            new_mode = 2;
                        }
                    }

                    if (temp_0_change){
                        if (num_per_grp != 1){
                            for (i=0; i<count; i++)
                                fill_formula(new_formula[i], f_struct, new_idx,
                                             new_temp_0+i*rows, temp_op_new, new_temp_1+i*rows,
                                             new_mode, add_sub, mul_div);
                        } else {
                            for (i=0; i<count; i++)
                                fill_formula(new_formula[i], f_struct, new_idx,
                                             new_temp_0+i*rows, temp_op_new, temp_1,
                                             new_mode, add_sub, mul_div);
                        }
                    } else {
                        if (num_per_grp != 1){
                            for (i=0; i<count; i++)
                                fill_formula(new_formula[i], f_struct, new_idx,
                                             temp_0, temp_op_new, new_temp_1+i*rows,
                                             new_mode, add_sub, mul_div);
                        } else {
                            for (i=0; i<count; i++)
                                fill_formula(new_formula[i], f_struct, new_idx,
                                             temp_0, temp_op_new, temp_1,
                                             new_mode, add_sub, mul_div);
                        }
                    }
                } else {
                    // cout << "XYZ" << endl;
                    cudaError_t status = cudaMemcpy(
                        temp_weight_storage+count_temp_storage*rows,
                        new_temp_0, 8*count*rows, cudaMemcpyDeviceToDevice
                    );
                    // cout << cudaGetErrorString(status) << endl;
                    for (i=0; i<count; i++){
                        // cout << count_temp_storage + i << endl;
                        memcpy(temp_formula_storage[count_temp_storage+i],
                               new_formula[i], fml_shape);
                    }
                    count_temp_storage += count;
                    // cout << "Ahihi" << endl;
                    if (count_temp_storage >= __STORAGE_SIZE__){
                        replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                            temp_weight_storage, rows, count_temp_storage
                        );
                        cudaDeviceSynchronize();
                        compute_result();
                        count_temp_storage = 0;
                    }
                    if (status){
                        // cout << cudaGetErrorString(status) << endl;
                        throw runtime_error("Cuda bad status");
                    }
                }

                // Giải phóng bộ nhớ
                for (i=0; i<count; i++){
                    delete[] new_formula[i];
                }
                delete[] new_formula;
                delete[] valid;
                cudaFree(d_valid);
                if (temp_0_change) cudaFree(new_temp_0);
                if (num_per_grp != 1) cudaFree(new_temp_1);
            }

            // Giải phóng bộ nhớ
            delete[] valid_operand;
        }
    }

    void compute_result(){
        fill_thresholds<<<count_temp_storage*(index_length-2)/256 + 1, 256>>>(
            temp_weight_storage, d_threshold, INDEX, index_length, count_temp_storage, rows
        );
        cudaDeviceSynchronize();
        
        int num_threshold = 5*(index_length - 2);
        multi_invest_2<<<count_temp_storage*num_threshold/256 + 1, 256>>>(
            temp_weight_storage, d_threshold, d_result, count_temp_storage, num_threshold,
            rows, __NUM_CYCLE__, __INTEREST__, INDEX, PROFIT, SYMBOL, BOOL_ARG, index_length
        );
        cudaDeviceSynchronize();

        find_finals<<<count_temp_storage*__NUM_CYCLE__*2/256+1, 256>>>(
            d_result, d_threshold, d_final, count_temp_storage, num_threshold, __NUM_CYCLE__
        );
        cudaDeviceSynchronize();

        //
        cudaMemcpy(h_final, d_final, 32*count_temp_storage*__NUM_CYCLE__, cudaMemcpyDeviceToHost);

        int i, j, k;
        for (i=0; i<count_temp_storage; i++){
            for (j=0; j<__NUM_CYCLE__; j++){
                if (h_final[i*__NUM_CYCLE__*4 + j*4 + 3] >= 1.37){
                    write_formula(temp_formula_storage[i], fml_shape/2, array_outFile[j]);
                    for (k=0; k<4; k++)
                        array_outFile[j] << h_final[i*__NUM_CYCLE__*4 + j*4 + k] << " ";
                    array_outFile[j] << endl;
                }
            }
        }
    }

    // Test
    void test_run(int num_opr_per_fml){
        fml_shape = num_opr_per_fml * 2;
        int i;
        for (i=0; i<__STORAGE_SIZE__+cols; i++) temp_formula_storage[i] = new uint8_t[fml_shape];
        uint8_t *formula = new uint8_t[fml_shape];
        double *temp_0, *temp_1;
        cudaMalloc((void**)&temp_0, 8*rows);
        cudaMalloc((void**)&temp_1, 8*rows);

        double *h_temp_0 = new double[rows];
        for (i=0; i<rows; i++) h_temp_0[i] = 0;
        delete[] current[0];
        current[0] = new int[2*num_opr_per_fml];
        current[1][0] = -1;
        current[2][0] = 0;

        for (i=0; i<__NUM_CYCLE__; i++)
            array_outFile[i].open(to_string(i)+".txt");

        for (int num_per_grp=1; num_per_grp<=num_opr_per_fml; num_per_grp++){
            if (num_opr_per_fml%num_per_grp) continue;
            this->num_per_grp = num_per_grp;
            groups = num_opr_per_fml / num_per_grp;
            int **f_struct = new int*[groups];
            for (i=0; i<groups; i++){
                f_struct[i] = new int[4];
                f_struct[i][0] = 0;
                f_struct[i][1] = num_per_grp;
                f_struct[i][2] = 1 + 2*num_per_grp*i;
                f_struct[i][3] = 0;
            }

            current[1][0]++;
            for (i=0; i<2*num_opr_per_fml; i++){
                formula[i] = 0;
                current[0][i] = 0;
            }

            cudaMemcpy(temp_0, h_temp_0, 8*rows, cudaMemcpyHostToDevice);
            cudaMemcpy(temp_1, h_temp_0, 8*rows, cudaMemcpyHostToDevice);
            fill_formula(formula, f_struct, 0, temp_0, 0, temp_1, 0, false, false);
            // cout << count_temp_storage << endl;
            replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                temp_weight_storage, rows, count_temp_storage
            );
            cudaDeviceSynchronize();
            compute_result();
            count_temp_storage = 0;

            // Giải phóng bộ nhớ
            for (i=0; i<groups; i++) delete[] f_struct[i];
            delete[] f_struct;
        }

        // Giải phóng bộ nhớ
        delete[] formula;
        delete[] h_temp_0;
        cudaFree(temp_0);
        cudaFree(temp_1);
        for (i=0; i<__STORAGE_SIZE__+cols; i++) delete[] temp_formula_storage[i];
    }
};
