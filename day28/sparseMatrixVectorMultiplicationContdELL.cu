#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

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
        if (sparse_m[i] != 0)
            (*non_zero)++;
    }

    cudaMallocManaged(cooRow, (*non_zero) * sizeof(int));
    cudaMallocManaged(cooCol, (*non_zero) * sizeof(int));
    cudaMallocManaged(cooVal, (*non_zero) * sizeof(int));

    cusparseSpMatDescr_t matCOO;
    cusparseCreateCoo(&matCOO, N, N, *non_zero, *cooRow, *cooCol, *cooVal,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I);

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
        if (sparse_m[i] != 0)
            (*non_zero)++;
    }

    cudaMallocManaged(csrRowPtr, (N + 1) * sizeof(int));
    cudaMallocManaged(csrCol, (*non_zero) * sizeof(int));
    cudaMallocManaged(csrVal, (*non_zero) * sizeof(int));

    cusparseSpMatDescr_t matCSR;
    cusparseCreateCsr(&matCSR, N, N, *non_zero, *csrRowPtr, *csrCol, *csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I);

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

void ELL(int *sparse_m, int **ellCol, int **ellVal, int *max_nnz) {
    int *row_nnz = (int*) malloc(N * sizeof(int));
    *max_nnz = 0;
    for (int i = 0; i < N; i++) 
    {
        int count = 0;
        for (int j = 0; j < N; j++) 
        {
            if (sparse_m[i * N + j] != 0)
                count++;
        }
        row_nnz[i] = count;
        if (count > *max_nnz)
            *max_nnz = count;
    }

    cudaMallocManaged(ellVal, N * (*max_nnz) * sizeof(int));
    cudaMallocManaged(ellCol, N * (*max_nnz) * sizeof(int));

    for (int i = 0; i < N * (*max_nnz); i++) 
    {
        (*ellVal)[i] = 0;
        (*ellCol)[i] = -1;
    }

    for (int i = 0; i < N; i++)
    {
        int count = 0;
        for (int j = 0; j < N; j++) 
        {
            int val = sparse_m[i * N + j];
            if (val != 0) 
            {
                int index = i * (*max_nnz) + count;
                (*ellVal)[index] = val;
                (*ellCol)[index] = j;
                count++;
            }
        }
    }
    free(row_nnz);
}

struct row_data 
{
    int index;
    int nnz;
};

int cmp_row_data(const void *a, const void *b) 
{
    const struct row_data *ra = (const struct row_data*) a;
    const struct row_data *rb = (const struct row_data*) b;
    return rb->nnz - ra->nnz;
}

void JDS(int *sparse_m, int **jdsRow, int **jdsCol, int **jdsVal, int **jdsPerm, int64_t *non_zero, int *max_nnz) 
{
    int *row_nnz = (int*) malloc(N * sizeof(int));
    int *perm = (int*) malloc(N * sizeof(int));
    *non_zero = 0;

    for (int i = 0; i < N; i++) 
    {
        int count = 0;
        for (int j = 0; j < N; j++) 
        {
            if (sparse_m[i * N + j] != 0)
                count++;
        }
        row_nnz[i] = count;
        perm[i] = i;
        *non_zero += count;
    }

    struct row_data *rows = (struct row_data*) malloc(N * sizeof(struct row_data));
    for (int i = 0; i < N; i++) 
    {
        rows[i].index = i;
        rows[i].nnz = row_nnz[i];
    }
    qsort(rows, N, sizeof(struct row_data), cmp_row_data);
    for (int i = 0; i < N; i++) 
    {
        perm[i] = rows[i].index;
    }

    int jds_max = (N > 0 ? rows[0].nnz : 0);
    *max_nnz = jds_max;

    cudaMallocManaged(jdsRow, (jds_max + 1) * sizeof(int));
    cudaMallocManaged(jdsCol, (*non_zero) * sizeof(int));
    cudaMallocManaged(jdsVal, (*non_zero) * sizeof(int));
    cudaMallocManaged(jdsPerm, N * sizeof(int));

    for (int i = 0; i < N; i++) 
    {
        (*jdsPerm)[i] = perm[i];
    }

    int idx = 0;
    (*jdsRow)[0] = 0;
    for (int d = 0; d < jds_max; d++) 
    {
        int count = 0;
        for (int i = 0; i < N; i++) 
        {
            if (row_nnz[perm[i]] > d)
                count++;
            else
                break;
        }
        idx += count;
        (*jdsRow)[d+1] = idx;
    }

    int pos = 0;
    for (int d = 0; d < jds_max; d++) 
    {
        for (int i = 0; i < N; i++) 
        {
            int row = perm[i];
            if (row_nnz[row] > d) 
            {
                int count = 0;
                for (int j = 0; j < N; j++) 
                {
                    if (sparse_m[row * N + j] != 0) 
                    {
                        if (count == d) 
                        {
                            (*jdsVal)[pos] = sparse_m[row * N + j];
                            (*jdsCol)[pos] = j;
                            pos++;
                            break;
                        }
                        count++;
                    }
                }
            } 
            else 
            {
                break;
            }
        }
    }
    free(row_nnz);
    free(perm);
    free(rows);
}

__global__ void spMV_COO(int *cooRow, int *cooCol, int *cooVal, int *in_vect, int *out_vect, int64_t non_zero)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < non_zero)
    {
        int row = cooRow[tid];
        int col = cooCol[tid];
        int val = cooVal[tid];
        atomicAdd(&out_vect[row], val * in_vect[col]);
    }
}

__global__ void spMV_CSR(int *csrRowPtr, int *csrCol, int *csrVal, int *in_vect, int *out_vect, int num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows)
    {
        int sum = 0;
        for (int i = csrRowPtr[tid]; i < csrRowPtr[tid + 1]; i++)
        {
            int col = csrCol[i];
            int val = csrVal[i];
            sum += in_vect[col] * val;
        }
        out_vect[tid] = sum;
    }
}

__global__ void spMV_ELL(int *ellCol, int *ellVal, int *in_vect, int *out_vect, int ell_max_nnz)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N)
    {
        int sum = 0;
        for (int d = 0; d < ell_max_nnz; d++)
        {
            int idx = row * ell_max_nnz + d;
            int col = ellCol[idx];
            int val = ellVal[idx];
            if (col != -1)
            {
                sum += val * in_vect[col];
            }
        }
        out_vect[row] = sum;
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
    int *sparse_m, *in_vect;
    cudaMallocManaged(&sparse_m, N * N * sizeof(int));
    cudaMallocManaged(&in_vect, N * sizeof(int));

    int *out_vect_coo, *out_vect_csr, *out_vect_ell, *out_vect_cpu;
    cudaMallocManaged(&out_vect_coo, N * sizeof(int));
    cudaMallocManaged(&out_vect_csr, N * sizeof(int));
    cudaMallocManaged(&out_vect_ell, N * sizeof(int));
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

    int *cooRow, *cooCol, *cooVal;
    int64_t non_zero_coo = 0;
    clock_t start_coo_creation = clock();
    COO(sparse_m, &cooRow, &cooCol, &cooVal, &non_zero_coo);
    clock_t end_coo_creation = clock();
    double coo_creation_time = ((double)(end_coo_creation - start_coo_creation)) / CLOCKS_PER_SEC * 1000.0;

    cudaEvent_t start, stop;
    float coo_operation_time;
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

    int *csrRowPtr, *csrCol, *csrVal;
    int64_t non_zero_csr = 0;
    clock_t start_csr_creation = clock();
    CSR(sparse_m, &csrRowPtr, &csrCol, &csrVal, &non_zero_csr);
    clock_t end_csr_creation = clock();
    double csr_creation_time = ((double)(end_csr_creation - start_csr_creation)) / CLOCKS_PER_SEC * 1000.0;

    int num_rows_csr = sizeof(csrRowPtr) / sizeof(csrRowPtr[0]);
    cudaEvent_t start2, stop2;
    float csr_operation_time;
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

    int *ellCol, *ellVal, ell_max_nnz;
    clock_t start_ell_creation = clock();
    ELL(sparse_m, &ellCol, &ellVal, &ell_max_nnz);
    clock_t end_ell_creation = clock();
    double ell_creation_time = ((double)(end_ell_creation - start_ell_creation)) / CLOCKS_PER_SEC * 1000.0;

    cudaEvent_t start3, stop3;
    float ell_operation_time;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    spMV_ELL<<<((N + THREADS - 1) / THREADS), THREADS>>>(ellCol, ellVal, in_vect, out_vect_ell, ell_max_nnz);
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&ell_operation_time, start3, stop3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    
    int *jdsRow, *jdsCol, *jdsVal, *jdsPerm;
    int64_t non_zero_jds = 0;
    int jds_max_nnz;
    clock_t start_jds_creation = clock();
    JDS(sparse_m, &jdsRow, &jdsCol, &jdsVal, &jdsPerm, &non_zero_jds, &jds_max_nnz);
    clock_t end_jds_creation = clock();
    double jds_creation_time = ((double)(end_jds_creation - start_jds_creation)) / CLOCKS_PER_SEC * 1000.0;
    
    double coo_total_time = coo_creation_time + coo_operation_time;
    double csr_total_time = csr_creation_time + csr_operation_time;
    double ell_total_time = ell_creation_time + ell_operation_time;

    printf("Dense Matrix Storage Size: %.2lf MB\n", (N * N * sizeof(int) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (COO): %.2lf MB\n", ((non_zero_coo * 3 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (CSR): %.2lf MB\n", (((N + 1) * sizeof(int) + non_zero_csr * 2 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (ELL): %.2lf MB\n", ((N * ell_max_nnz * 2 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (JDS): %.2lf MB\n\n", ((((jds_max_nnz + 1) * sizeof(int)) + non_zero_jds * 2 * sizeof(int) + N * sizeof(int)) / (1024.0 * 1024)));
    
    printf("COO Creation Time: %f ms\n", coo_creation_time);
    printf("COO Operation Time: %f ms\n", coo_operation_time);
    printf("COO Total Time: %f ms\n\n", coo_total_time);
    
    printf("CSR Creation Time: %f ms\n", csr_creation_time);
    printf("CSR Operation Time: %f ms\n", csr_operation_time);
    printf("CSR Total Time: %f ms\n\n", csr_total_time);

    printf("ELL Creation Time: %f ms\n", ell_creation_time);
    printf("ELL Operation Time: %f ms\n", ell_operation_time);
    printf("ELL Total Time: %f ms\n\n", ell_total_time);
    
    printf("CPU spMV Time: %f ms\n\n", cpu_time);
    
    cudaFree(sparse_m);
    cudaFree(in_vect);
    cudaFree(out_vect_coo);
    cudaFree(out_vect_csr);
    cudaFree(out_vect_cpu);
    cudaFree(cooRow);
    cudaFree(cooCol);
    cudaFree(cooVal);
    cudaFree(csrRowPtr);
    cudaFree(csrCol);
    cudaFree(csrVal);
    cudaFree(ellCol);
    cudaFree(ellVal);
    cudaFree(jdsRow);
    cudaFree(jdsCol);
    cudaFree(jdsVal);
    cudaFree(jdsPerm);
    
    return 0;
}
