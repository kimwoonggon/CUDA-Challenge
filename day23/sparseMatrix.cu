#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include <cusparse_v2.h>

#define N (1<<12)
#define THREADS 1024
#define BLOCKS ((N * N + THREADS - 1) / THREADS)
#define SPARSITY 0.1f

__global__ void init(int *sparse_m, unsigned long seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = N * N;
    if (tid < totalElements) 
    {
        curandState state;
        curand_init(seed, tid, 0, &state);
        float rand_val = curand_uniform(&state);
        sparse_m[tid] = (rand_val <= SPARSITY) ? (curand(&state) % 100 + 1) : 0;
    }
}

void COO(int *sparse_m, int **cooRow, int **cooCol, int **cooVal, int64_t *non_zero) 
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseDnMatDescr_t matrix;
    cusparseCreateDnMat(&matrix, N, N, N, sparse_m, CUDA_R_32I, CUSPARSE_ORDER_ROW);

    *non_zero = 0;
    for (int i = 0; i < N * N; i++) 
    {
        if (sparse_m[i] != 0) (*non_zero)++;
    }

    cudaMallocManaged(cooRow, (*non_zero) * sizeof(int));
    cudaMallocManaged(cooCol, (*non_zero) * sizeof(int));
    cudaMallocManaged(cooVal, (*non_zero) * sizeof(int));

    cusparseSpMatDescr_t matCOO;
    cusparseCreateCoo(&matCOO, N, N, *non_zero, *cooRow, *cooCol, *cooVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I);

    size_t bufferSize = 0;
    void *dBuffer = NULL;
    cusparseDenseToSparseAlg_t alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    
    cusparseDenseToSparse_bufferSize(handle, matrix, matCOO, alg, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseDenseToSparse_analysis(handle, matrix, matCOO, alg, &bufferSize);
    cusparseDenseToSparse_convert(handle, matrix, matCOO, alg, dBuffer);

    cudaFree(dBuffer);
    cusparseDestroyDnMat(matrix);
    cusparseDestroySpMat(matCOO);
    cusparseDestroy(handle);
}

void CSR(int *sparse_m, int **csrRowPtr, int **csrCol, int **csrVal, int64_t *non_zero) 
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseDnMatDescr_t matrix;
    cusparseCreateDnMat(&matrix, N, N, N, sparse_m, CUDA_R_32I, CUSPARSE_ORDER_ROW);

    *non_zero = 0;
    for (int i = 0; i < N * N; i++) 
    {
        if (sparse_m[i] != 0) (*non_zero)++;
    }

    cudaMallocManaged(csrRowPtr, (N + 1) * sizeof(int));
    cudaMallocManaged(csrCol, (*non_zero) * sizeof(int));
    cudaMallocManaged(csrVal, (*non_zero) * sizeof(int));

    cusparseSpMatDescr_t matCSR;
    cusparseCreateCsr(&matCSR, N, N, *non_zero, *csrRowPtr, *csrCol, *csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I);

    size_t bufferSize = 0;
    void *dBuffer = NULL;
    cusparseDenseToSparseAlg_t alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    
    cusparseDenseToSparse_bufferSize(handle, matrix, matCSR, alg, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseDenseToSparse_analysis(handle, matrix, matCSR, alg, &bufferSize);
    cusparseDenseToSparse_convert(handle, matrix, matCSR, alg, dBuffer);

    cudaFree(dBuffer);
    cusparseDestroyDnMat(matrix);
    cusparseDestroySpMat(matCSR);
    cusparseDestroy(handle);
}

int main() 
{
    int *sparse_m, *cooRow, *cooCol, *cooVal, *csrRowPtr, *csrCol, *csrVal;
    int64_t non_zero_coo = 0, non_zero_csr = 0;

    cudaMallocManaged(&sparse_m, N * N * sizeof(int));

    init<<<BLOCKS, THREADS>>>(sparse_m, time(NULL));
    cudaDeviceSynchronize();

    COO(sparse_m, &cooRow, &cooCol, &cooVal, &non_zero_coo);

    CSR(sparse_m, &csrRowPtr, &csrCol, &csrVal, &non_zero_csr);

    printf("Dense Matrix Storage Size: %.2lf MB\n", (N * N * sizeof(int) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (COO): %.2lf MB\n", ((non_zero_coo * 3 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (CSR): %.2lf MB\n", (((N + 1) * sizeof(int) + non_zero_csr * 2 * sizeof(int)) / (1024.0 * 1024)));

    cudaFree(sparse_m);
    cudaFree(cooRow);
    cudaFree(cooCol);
    cudaFree(cooVal);
    cudaFree(csrRowPtr);
    cudaFree(csrCol);
    cudaFree(csrVal);

    return 0;
}
