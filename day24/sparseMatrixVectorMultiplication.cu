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
    if (tid < N * N) 
    {
        curandState state;
        curand_init(seed, tid, 0, &state);
        float rand_val = curand_uniform(&state);
        sparse_m[tid] = (rand_val <= SPARSITY) ? (curand(&state) % 100 + 1) : 0;
    }
}

__global__ void initVect(int *vect, unsigned long seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) 
    {
        curandState state;
        curand_init(seed, tid, 0, &state);
        float rand_val = curand_uniform(&state);
        vect[tid] = curand(&state) % 100 + 1;
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

__global__ void spMV_COO(int *cooRow, int *cooCol, int *cooVal, int *in_vect, int *out_vect, int64_t non_zero)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < non_zero)
    {
        int row = cooRow[tid];
        int col = cooCol[tid];
        int val = cooVal[tid];
        atomicAdd(&out_vect[row], val * in_vect[col]);
    }
}

__global__ void spMV_CSR(int *csrRowPtr, int *csrCol, int *csrVal, int *in_vect, int *out_vect, int num_rows_csr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < num_rows_csr)
    {
        int sum = 0;
        for(int i = csrRowPtr[tid]; i < csrRowPtr[tid + 1]; i++)
        {
            int col = csrCol[i];
            int val = csrVal[i];
            sum += in_vect[col] * val;
        }
        out_vect[tid] = sum;
    }
}

void spMV_CPU(int *sparse_m, int *in_vect, int *out_vect)
{
    for (int row = 0; row < N; row++) 
    {
        int sum = 0;
        for (int col = 0; col < N; col++) 
        {
            sum += sparse_m[row * N + col] * in_vect[col];
        }
        out_vect[row] = sum;
    }
}


int main() 
{
    int *sparse_m, *cooRow, *cooCol, *cooVal, *csrRowPtr, *csrCol, *csrVal, *in_vect, *out_vect_coo, *out_vect_csr, *out_vect_cpu;
    int64_t non_zero_coo = 0, non_zero_csr = 0;
    float coo_operation_time, csr_operation_time;
    double coo_creation_time, csr_creation_time;
    double coo_total_time, csr_total_time;

    cudaMallocManaged(&sparse_m, N * N * sizeof(int));
    cudaMallocManaged(&in_vect, N * sizeof(int));
    cudaMallocManaged(&out_vect_coo, N * sizeof(int));
    cudaMallocManaged(&out_vect_csr, N * sizeof(int));
    cudaMallocManaged(&out_vect_cpu, N * sizeof(int));

    cudaMemset(out_vect_coo, 0, N * sizeof(int));
    cudaMemset(out_vect_csr, 0, N * sizeof(int));
    cudaMemset(out_vect_cpu, 0, N * sizeof(int));

    init<<<BLOCKS, THREADS>>>(sparse_m, time(NULL));
    cudaDeviceSynchronize();

    initVect<<<((N + THREADS - 1)/THREADS), THREADS>>>(in_vect, time(NULL));
    cudaDeviceSynchronize();

    clock_t cpu_start = clock();
    spMV_CPU(sparse_m, in_vect, out_vect_cpu);
    float cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    clock_t start_coo_creation = clock();
    COO(sparse_m, &cooRow, &cooCol, &cooVal, &non_zero_coo);
    clock_t end_coo_creation = clock();
    coo_creation_time = ((double)(end_coo_creation - start_coo_creation)) / CLOCKS_PER_SEC * 1000.0;

    clock_t start_csr_creation = clock();
    CSR(sparse_m, &csrRowPtr, &csrCol, &csrVal, &non_zero_csr);
    clock_t end_csr_creation = clock();
    csr_creation_time = ((double)(end_csr_creation - start_csr_creation)) / CLOCKS_PER_SEC * 1000.0;

    int num_rows_csr = sizeof(csrRowPtr) / sizeof(csrRowPtr[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    spMV_COO<<<((non_zero_coo + THREADS - 1) / THREADS), THREADS>>>(cooRow, cooCol, cooVal, in_vect, out_vect_coo, non_zero_coo);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&coo_operation_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    spMV_CSR<<<((num_rows_csr + THREADS - 1) / THREADS), THREADS>>>(csrRowPtr, csrCol, csrVal, in_vect, out_vect_csr, num_rows_csr);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&csr_operation_time, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    coo_total_time = coo_creation_time + coo_operation_time;
    csr_total_time = csr_creation_time + csr_operation_time;

    printf("Dense Matrix Storage Size: %.2lf MB\n", (N * N * sizeof(int) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (COO): %.2lf MB\n", ((non_zero_coo * 3 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (CSR): %.2lf MB\n\n", (((N + 1) * sizeof(int) + non_zero_csr * 2 * sizeof(int)) / (1024.0 * 1024)));

    printf("COO Creation Time: %f ms\n", coo_creation_time);
    printf("COO Operation Time: %f ms\n", coo_operation_time);
    printf("COO Total Time: %f ms\n\n", coo_total_time);

    printf("CSR Creation Time: %f ms\n", csr_creation_time);
    printf("CSR Operation Time: %f ms\n", csr_operation_time);
    printf("CSR Total Time: %f ms\n\n", csr_total_time);

    printf("CPU spMV Time: %f ms\n", cpu_time);

    cudaFree(sparse_m);
    cudaFree(cooRow);
    cudaFree(cooCol);
    cudaFree(cooVal);
    cudaFree(csrRowPtr);
    cudaFree(csrCol);
    cudaFree(csrVal);
    cudaFree(in_vect);
    cudaFree(out_vect_coo);
    cudaFree(out_vect_csr);
    cudaFree(out_vect_cpu);

    return 0;
}
