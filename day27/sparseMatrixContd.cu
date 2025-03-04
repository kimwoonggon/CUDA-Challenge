#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include <cusparse_v2.h>
#include <stdlib.h>

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

int main() 
{
    int *sparse_m, *cooRow, *cooCol, *cooVal, *csrRowPtr, *csrCol, *csrVal;
    int *ellCol, *ellVal;
    int ell_max_nnz;
    int *jdsRow, *jdsCol, *jdsVal, *jdsPerm;
    int64_t non_zero_coo = 0, non_zero_csr = 0, non_zero_jds = 0;
    int jds_max_nnz;

    cudaMallocManaged(&sparse_m, N * N * sizeof(int));

    init<<<BLOCKS, THREADS>>>(sparse_m, time(NULL));
    cudaDeviceSynchronize();

    COO(sparse_m, &cooRow, &cooCol, &cooVal, &non_zero_coo);
    CSR(sparse_m, &csrRowPtr, &csrCol, &csrVal, &non_zero_csr);
    ELL(sparse_m, &ellCol, &ellVal, &ell_max_nnz);
    JDS(sparse_m, &jdsRow, &jdsCol, &jdsVal, &jdsPerm, &non_zero_jds, &jds_max_nnz);

    printf("Dense Matrix Storage Size: %.2lf MB\n", (N * N * sizeof(int) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (COO): %.2lf MB\n", ((non_zero_coo * 3 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (CSR): %.2lf MB\n", (((N + 1) * sizeof(int) + non_zero_csr * 2 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (ELL): %.2lf MB\n", ((N * ell_max_nnz * 2 * sizeof(int)) / (1024.0 * 1024)));
    printf("Sparse Matrix Storage Size (JDS): %.2lf MB\n",((((jds_max_nnz + 1) * sizeof(int)) + non_zero_jds * 2 * sizeof(int) + N * sizeof(int)) / (1024.0 * 1024)));

    cudaFree(sparse_m);
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
