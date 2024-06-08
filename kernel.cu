#include <iostream>
#include <cuda_runtime.h>


__device__ const double __NEGATIVE_INFINITY__ = -1.7976931348623157e+308;
__device__ const double __POSITIVE_INFINITY__ = +1.7976931348623157e+308;


__global__ void cuda_array_assign(
    double* array,
    int length,
    double value
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) array[index] = value;
}


__global__ void copy_from_operands(
    double *dest,
    double *operands,
    int *arrCpy,
    int length,
    int numCpy
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numCpy*length){
        int i = index / length;
        int j = index % length;
        dest[index] = operands[arrCpy[i]*length + j];
    }
}


__global__ void update_temp_weight(
    double *temp_weight_new,
    double *temp_weight_old,
    double *operands,
    int *arrOpr,
    int length,
    int numOpr,
    bool isMul
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isMul)
            temp_weight_new[index] = temp_weight_old[j] * operands[arrOpr[i]*length + j];
        else
            temp_weight_new[index] = temp_weight_old[j] / operands[arrOpr[i]*length + j];
    }
}


__global__ void update_last_weight(
    double *last_weight,
    double *curr_weight,
    double *temp_weight,
    int length,
    int numOpr,
    bool isAdd
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + temp_weight[index];
        else last_weight[index] = curr_weight[j] - temp_weight[index];
    }
}


__global__ void update_last_weight_through_operands(
    double *last_weight,
    double *curr_weight,
    double *operands,
    int *arrOpr,
    int length,
    int numOpr,
    bool isAdd
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + operands[arrOpr[i]*length + j];
        else last_weight[index] = curr_weight[j] - operands[arrOpr[i]*length + j];
    }
}


__global__ void replace_nan_and_inf(
    double *array,
    int length,
    int numOpr
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        if (isnan(array[index]) || isinf(array[index]))
            array[index] = __NEGATIVE_INFINITY__;
    }
}


__device__ double max_of_array(
    double *array,
    int left,
    int right,
    double supremum
) {
    double max_ = __NEGATIVE_INFINITY__;
    for (int i=left; i<right; i++){
        if (array[i] < supremum && array[i] > max_) max_ = array[i];
    }
    return max_;
}


__device__ void top_n_of_array(
    double *array,
    int left,
    int right,
    double *result,
    int start,
    int n
) {
    double supremum = __POSITIVE_INFINITY__;
    for (int i=0; i<n; i++){
        supremum = max_of_array(array, left, right, supremum);
        result[start+i] = supremum;
    }
}


__global__ void fill_thresholds(
    double *weights,
    double *thresholds,
    int *INDEX,
    int index_length,
    int num_array,
    int length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cycle = index_length - 2;
    if (index < num_array*num_cycle){
        int ix = index % num_cycle;
        int iy = index / num_cycle;
        top_n_of_array(weights + iy*length,
                       INDEX[ix+1], INDEX[ix+2],
                       thresholds + iy*5*num_cycle,
                       ix*5, 5);
    }
}


__device__ void _multi_invest_2(
    double *weight,
    double threshold,
    int t_idx,
    double *result,
    double INTEREST,
    int *INDEX,
    double *PROFIT,
    int *SYMBOL,
    int *BOOL_ARG,
    int index_size,
    int num_cycle
) {
    int reason = 0;
    double Geo2 = 0, Har2 = 0;
    int start, end, end2, count, k, sym, s, rs_idx;
    double temp, n;
    bool check;
    for (int i=index_size-3; i>0; i--){
        start = INDEX[i];
        end = INDEX[i+1];
        temp = 0;
        count = 0;
        check = false;
        if (!reason){
            end2 = INDEX[i+2];
            for (k=start; k<end; k++){
                if (weight[k] > threshold && BOOL_ARG[k]){
                    check = true;
                    sym = SYMBOL[k];
                    for (s=end; s<end2; s++){
                        if (SYMBOL[s] == sym){
                            if (weight[s] > threshold){
                                count++;
                                temp += PROFIT[k];
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            for (k=start; k<end; k++){
                if (weight[k] > threshold && BOOL_ARG[k]){
                    check = true;
                    count++;
                    temp += PROFIT[k];
                }
            }
        }

        if (!count){
            Geo2 += log(INTEREST);
            Har2 += 1.0 / INTEREST;
            if (!check) reason = 1;
        } else {
            temp /= count;
            Geo2 += log(temp);
            Har2 += 1.0 / temp;
            reason = 0;
        }

        if (i <= num_cycle && t_idx+1 >= i){
            rs_idx = num_cycle - i;
            n = index_size - 2 - i;
            result[2*rs_idx] = exp(Geo2/n);
            result[2*rs_idx+1] = n / Har2;
        }
    }
}


__global__ void multi_invest_2(
    double *weights,
    double *thresholds,
    double *results,
    int num_array,
    int num_threshold,
    int length,
    int num_cycle,
    double INTEREST,
    int *INDEX,
    double *PROFIT,
    int *SYMBOL,
    int *BOOL_ARG,
    int index_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_array*num_threshold){
        int ix = index % num_threshold;
        int iy = index / num_threshold;
        _multi_invest_2(
            weights + iy*length,
            thresholds[iy*num_threshold + ix],
            ix / 5,
            results + iy*num_threshold*num_cycle*2 + ix*num_cycle*2,
            INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG, index_size, num_cycle
        );
    }
}


__global__ void find_finals(
    double *results,
    double *thresholds,
    double *finals,
    int num_array,
    int num_threshold,
    int num_cycle
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 2*num_array*num_cycle){
        int iz = index % 2;
        int ix = (index/2) % num_cycle;
        int iy = (index/2) / num_cycle;

        double *result = results + iy*num_threshold*num_cycle*2;
        double *threshold = thresholds + iy*num_threshold;
        double *final_ = finals + iy*num_cycle*4 + ix*4;

        final_[2*iz] = threshold[0];
        final_[2*iz + 1] = result[2*ix + iz];
        for (int i=1; i<num_threshold; i++){
            if (result[i*num_cycle*2 + 2*ix + iz] > final_[2*iz + 1]){
                final_[2*iz] = threshold[i];
                final_[2*iz + 1] = result[i*num_cycle*2 + 2*ix + iz];
            }
        }
    }
}
