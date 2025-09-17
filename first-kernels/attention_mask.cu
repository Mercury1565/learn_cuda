#include <cuda_runtime.h>
#include <stdio.h>

// Device function to apply mask
__device__ float apply_mask(float value, float mask) {
    return value * mask;
}

// Kernel to apply mask to scores
__global__ void attention_kernel(float *d_scores, float *d_mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_scores[idx] = apply_mask(d_scores[idx], d_mask[idx]);
    }
}

int main() {
    // Array size
    int n = 8;
    int size = n * sizeof(float);

    // Host arrays
    float h_scores[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_mask[] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    float *h_result = new float[n];

    // Device arrays
    float *d_scores, *d_mask;
    cudaMalloc(&d_scores, size);
    cudaMalloc(&d_mask, size);

    // Copy to device
    cudaMemcpy(d_scores, h_scores, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 4;  // Small for demo
    int grid_size = (n + block_size - 1) / block_size;  // Ceiling division: 2 blocks
    attention_kernel<<<grid_size, block_size>>>(d_scores, d_mask, n);
    cudaDeviceSynchronize();  // Wait for kernel to finish

    // Copy result back
    cudaMemcpy(h_result, d_scores, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Scores: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_scores[i]);
    printf("\nMask:   ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_mask[i]);
    printf("\nResult: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_result[i]);
    printf("\n");

    // Expected output:
    // Scores: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
    // Mask:   1.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0
    // Result: 1.0 0.0 3.0 0.0 5.0 6.0 0.0 8.0

    // Cleanup
    cudaFree(d_scores);
    cudaFree(d_mask);
    delete[] h_result;
    return 0;
}