#include <cuda_runtime.h>
#include <stdio.h>

// Device function to apply mask
__device__ float apply_mask(float value, float mask) {
    return value * mask;
}

// Kernel to apply mask to 2D matrix
__global__ void attention_kernel_2d(float *d_scores, float *d_mask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x < width && y < height) {
        d_scores[idx] = apply_mask(d_scores[idx], d_mask[idx]);
    }
}

int main() {
    // Matrix dimensions
    int n = 4;
    int size = n * n * sizeof(float);

    // Host arrays (2D)
    float h_scores[n][n] = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };
    float h_mask[n][n] = {
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0, 0.0},
        {1.0, 0.0, 0.0, 1.0}
    };
    float h_result[n][n]; // 2D array for result

    // Device arrays
    float *d_scores, *d_mask;
    cudaMalloc(&d_scores, size);
    cudaMalloc(&d_mask, size);

    // Copy to device
    cudaMemcpy(d_scores, h_scores, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block_size(2, 2); // 2x2 threads per block
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    attention_kernel_2d<<<grid_size, block_size>>>(d_scores, d_mask, n, n);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_result, d_scores, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Scores:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%.1f ", h_scores[i][j]);
        printf("\n");
    }
    printf("Mask:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%.1f ", h_mask[i][j]);
        printf("\n");
    }
    printf("Result:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%.1f ", h_result[i][j]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_scores);
    cudaFree(d_mask);
    return 0;
}