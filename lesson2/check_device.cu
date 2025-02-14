#include <stdio.h>
#include <cuda_runtime.h> 

//cudaGetDeviceProperties真能在主机端使用，不能写在核函数中
// __global__ void check_device()
// {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0); // 0 表示第一张 GPU
//     printf("Device Name: %s\n", prop.name);
//     printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
// }
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 表示第一张 GPU
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
}