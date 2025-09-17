#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void attention_kernel(float *d_scores, float *d_mask, int *d_debug, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Direct write to test memory access
        volatile float *scores = d_scores; // Prevent optimization
        scores[idx] = d_mask[idx] * d_scores[idx]; // Simplified operation
        // Debug: Record thread execution
        d_debug[idx] = idx;
        // Sentinel values
        if (idx == 0) scores[0] = 999.0f;
        if (idx == 1) scores[1] = 888.0f;
    }
}

int main() {
    // Print CUDA device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Running on GPU: %s (Compute Capability: %d.%d)\n", prop.name, prop.major, prop.minor);

    int n = 8;
    int size = n * sizeof(float);
    int debug_size = n * sizeof(int);

    // Host arrays
    float h_scores[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_mask[] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    float *h_result = new float[n];
    int *h_debug = new int[n];

    // Device arrays
    float *d_scores, *d_mask;
    int *d_debug;
    CUDA_CHECK(cudaMalloc(&d_scores, size));
    CUDA_CHECK(cudaMalloc(&d_mask, size));
    CUDA_CHECK(cudaMalloc(&d_debug, debug_size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_debug, 0, debug_size)); // Initialize debug array

    // Verify device copy
    float *h_test = new float[n];
    CUDA_CHECK(cudaMemcpy(h_test, d_scores, size, cudaMemcpyDeviceToHost));
    printf("Test copy (before kernel): ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_test[i]);
    printf("\nTest mask copy: ");
    CUDA_CHECK(cudaMemcpy(h_test, d_mask, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) printf("%.1f ", h_test[i]);
    printf("\n");
    delete[] h_test;

    // Launch kernel
    int block_size = 4;
    int grid_size = (n + block_size - 1) / block_size;
    printf("Launching kernel with %d blocks, %d threads per block (%d total threads)\n",
           grid_size, block_size, grid_size * block_size);
    attention_kernel<<<grid_size, block_size>>>(d_scores, d_mask, d_debug, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_result, d_scores, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug, d_debug, debug_size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Scores: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_scores[i]);
    printf("\nMask:   ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_mask[i]);
    printf("\nResult: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_result[i]);
    printf("\nDebug:  ");
    for (int i = 0; i < n; i++) printf("%d ", h_debug[i]);
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_debug));
    delete[] h_result;
    delete[] h_debug;
    return 0;
}