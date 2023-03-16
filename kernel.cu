
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

enum ChampionType {
    ApothecaryType,
    WarcasterType,
    KymerType,
    SkullcrusherType,
    RenegateType,
    OtherType
};

struct Entity {
    int turnMeter = 0;
    int speed;
};

struct ChampionStruct : Entity {  // Structure declaration  
    char name[20];
    int skillCooldown;
    int skillDelay;
    int unkillableDuration;
    ChampionType type;
};

ChampionStruct* Apothecary() {
    ChampionStruct* result = new ChampionStruct();
    result->type = ChampionType::ApothecaryType;
    result->skillCooldown = 3;
    return result;
}


ChampionStruct* Warcaster() {
    ChampionStruct* result = new ChampionStruct();
    result->type = ChampionType::WarcasterType;
    result->skillCooldown = 4;
    return result;
}

ChampionStruct* Kymer() {
    ChampionStruct* result = new ChampionStruct();
    result->type = ChampionType::KymerType;
    result->skillCooldown = 6;
    return result;
}


ChampionStruct* Other() {
    ChampionStruct* result = new ChampionStruct();
    result->type = ChampionType::OtherType;
    return result;
}

struct ClanBoss : Entity {
    int turn;
};

struct Simulation {
    ChampionStruct c1, c2, c3, c4, c5;
    ClanBoss cb;
};

struct SimulationParams {
    int c1StartSpeed;
    int c1SpeedSteps;
    int c1SkillDelayMin;
    int c1SkillDelaySteps;

    int c2StartSpeed;
    int c2SpeedSteps;
    int c2SkillDelayMin;
    int c2SkillDelaySteps;

    int c3StartSpeed;
    int c3SpeedSteps;
    int c3SkillDelayMin;
    int c3SkillDelaySteps;

    int c4StartSpeed;
    int c4SpeedSteps;
    int c4SkillDelayMin;
    int c4SkillDelaySteps;

    int c5StartSpeed;
    int c5SpeedSteps;
    int c5SkillDelayMin;
    int c5SkillDelaySteps;
};

SimulationParams* GetSimulationParams() {
    SimulationParams* params = new SimulationParams;
    params->c1StartSpeed = 200;
    params->c1SpeedSteps = 150;
    params->c1SkillDelayMin = 0;
    params->c1SkillDelaySteps = 6;
    //----------
    params->c2StartSpeed = 200;
    params->c2SpeedSteps = 100;
    params->c2SkillDelayMin = 0;
    params->c2SkillDelaySteps = 6;
    //----------
    params->c3StartSpeed = 200;
    params->c3SpeedSteps = 100;
    params->c3SkillDelayMin = 0;
    params->c3SkillDelaySteps = 6;
    //----------
    params->c4StartSpeed = 200;
    params->c4SpeedSteps = 100;
    params->c4SkillDelayMin = 0;
    params->c4SkillDelaySteps = 0;
    //----------
    params->c5StartSpeed = 200;
    params->c5SpeedSteps = 100;
    params->c5SkillDelayMin = 0;
    params->c5SkillDelaySteps = 0;

    return params;
}

// N is the maximum number of structs to insert
#define N 10000

typedef struct {
    int A, B, C;
} Match;

__device__ Match dev_data[N];
__device__ int dev_count = 2;

__device__ int my_push_back(Match* mt) {
    int insert_pt = atomicAdd(&dev_count, 1);
    if (insert_pt < N) {
        dev_data[insert_pt] = *mt;
        return insert_pt;
    }
    else return -1;
}

uint32_t CalculateSimulationParamsVariations(SimulationParams* params) {
    return (params->c1SpeedSteps + 1) * (params->c1SkillDelaySteps + 1) *
        (params->c2SpeedSteps + 1) * (params->c2SkillDelaySteps + 1) *
        (params->c3SpeedSteps + 1) * (params->c3SkillDelaySteps + 1) *
        (params->c4SpeedSteps + 1) * (params->c4SkillDelaySteps + 1) *
        (params->c5SpeedSteps + 1) * (params->c5SkillDelaySteps + 1);
}

cudaError_t testWithCuda(Simulation* simulation, SimulationParams* params, int* result);

__global__ void tickTurnmeter(Entity* e) {
    e->turnMeter += e->speed;
}

__global__ void test(Simulation* simulation, SimulationParams* params) {
    int i = threadIdx.x;
    atomicAdd(&dev_count, 1);
  //  if (i == 1) {
   //     Match* m = new Match;
    //    m->A = threadIdx.x;
    //    my_push_back(m);
    //}
    /*
    result[0] = threadIdx.x;
    result[1] = blockIdx.x;
    result[2] = threadIdx.z;
    result[3] = blockDim.x;
    result[4] = blockDim.z;*/
   // tickTurnmeter(c1);
   // return c1->speed;
}


int main()
{
    const int arraySize = 5;
    int result[arraySize];

    Simulation* x = new Simulation;
    memcpy(&(x->c1), Warcaster(), sizeof(ChampionStruct));
    memcpy(&(x->c2), Apothecary(), sizeof(ChampionStruct));
    memcpy(&(x->c3), Kymer(), sizeof(ChampionStruct));
    memcpy(&(x->c4), Other(), sizeof(ChampionStruct));
    memcpy(&(x->c5), Other(), sizeof(ChampionStruct));

    SimulationParams* params = GetSimulationParams();



    

    // Add vectors in parallel.
    cudaError_t cudaStatus = testWithCuda(x, params, result);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "testWithCuda failed!");
        return 1;
    }


    //printf("{%d,%d,%d,%d,%d}\n",
    //    result[0], result[1], result[2], result[3], result[4]);

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
cudaError_t testWithCuda(Simulation* simulation, SimulationParams* params, int* result)
{
    SimulationParams* gpuSimulationParams;
    Simulation* gpuSimulation;
   // int *dev_b = 0;
    int *gpu_result = 0;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpuSimulationParams, sizeof(SimulationParams));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpuSimulation, sizeof(Simulation));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // TODO size
    cudaStatus = cudaMalloc((void**)&gpu_result, 5 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(gpuSimulationParams, params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(gpuSimulation, simulation, sizeof(Simulation), cudaMemcpyHostToDevice);
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

    uint32_t variations = CalculateSimulationParamsVariations(params);
    int block_size = 1024;
    int blocks_count = variations / block_size;
    fprintf(stderr, "CalculateSimulationParamsVariations %u\nblock_size=%i\nblocks_count=%i\n", variations, block_size, blocks_count);


   



    // Launch a kernel on the GPU with one thread for each element.
    test <<<blocks_count, block_size >>>(gpuSimulation, gpuSimulationParams);
    
    
    int dsize;
    cudaStatus = cudaMemcpyFromSymbol(&dsize, dev_count, sizeof(int));
    //cudaStatus = cudaMemcpy(&dsize, &dev_count, sizeof(int), cudaMemcpyDeviceToHost);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    printf("gpuResult.count=%d\n", dsize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, gpu_result, 5, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(gpuSimulation);
    cudaFree(gpuSimulationParams);
    
    return cudaStatus;
}
