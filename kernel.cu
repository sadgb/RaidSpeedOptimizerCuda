
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



enum color { red, yellow, green = 20, blue };

enum ChampionType {
    Apothecary,
    Warcaster,
    Kymer,
    Skullcrusher,
    Renegate
};

struct Entity {
    int turnMeter = 0;
    int speed;
};

struct ChampionStruct  {  // Structure declaration  
    int turnMeter = 0;
    int speed;
    char name[20];
    int skillCooldown;
    int skillDelay;
    int unkillableDuration;
    ChampionType type;
};

struct ClanBoss : Entity {
    int turn;
};

cudaError_t testWithCuda(ChampionStruct* champion1, const int* champion1_speeds, unsigned int champion1_speeds_size, int* result);

__global__ void tickTurnmeter(Entity* e) {
    e->turnMeter += e->speed;
}

__global__ void test(ChampionStruct* c1, const int *champion1Speeds, int * result ) {
    int i = threadIdx.x;

    result[i] = c1->turnMeter + champion1Speeds[i];
   // tickTurnmeter(c1);
   // return c1->speed;
}


int main()
{
    const int arraySize = 5;
    const int champion1_speeds[arraySize] = { 1, 2, 3, 4, 5 };
    int result[arraySize];

    ChampionStruct* x = new ChampionStruct;
    x->turnMeter = 1;

    // Add vectors in parallel.
    cudaError_t cudaStatus = testWithCuda(x, champion1_speeds, arraySize, result);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{%d,%d,%d,%d,%d}\n",
        result[0], result[1], result[2], result[3], result[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t testWithCuda(ChampionStruct* champion1, const int * champion1_speeds, unsigned int champion1_speeds_size, int * result)
{
    int *dev_a = 0;
    ChampionStruct* gpu_champion1;
   // int *dev_b = 0;
    int *gpu_result = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, champion1_speeds_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpu_champion1, sizeof(ChampionStruct));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_result, champion1_speeds_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, champion1_speeds, champion1_speeds_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_champion1, champion1, sizeof(ChampionStruct), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /*
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    */

    // Launch a kernel on the GPU with one thread for each element.
    test <<<1, champion1_speeds_size >>>(gpu_champion1, dev_a, gpu_result);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, gpu_result, champion1_speeds_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_champion1);
    cudaFree(dev_a);
    
    return cudaStatus;
}
