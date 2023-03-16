
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
    int speed30Duration = 0;
};

struct ChampionStruct : Entity {  // Structure declaration  
    char name[20];
    int skillCooldown = 0;
    int skillCurrentCooldown = 0;
    int skillDelay = 0;
    int unkillableDuration = 0;
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
    int turnesMade = 0;
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

    int cbSpeed = 190;
};


/*
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
/**/
/**/
// working example
// Fastest speed tuned team had speeds : 302, 286, 286, 270, 200
// delays : d1 = 1 d2 = 0 d3 = 2
SimulationParams* GetSimulationParams() {
    SimulationParams* params = new SimulationParams;
    params->c1StartSpeed = 302;
    params->c1SpeedSteps = 0;
    params->c1SkillDelayMin = 1;
    params->c1SkillDelaySteps = 0;
    //----------
    params->c2StartSpeed = 286;
    params->c2SpeedSteps = 0;
    params->c2SkillDelayMin = 0;
    params->c2SkillDelaySteps = 0;
    //----------
    params->c3StartSpeed = 286;
    params->c3SpeedSteps = 0;
    params->c3SkillDelayMin = 2;
    params->c3SkillDelaySteps = 0;
    //----------
    params->c4StartSpeed = 270;
    params->c4SpeedSteps = 0;
    params->c4SkillDelayMin = 0;
    params->c4SkillDelaySteps = 0;
    //----------
    params->c5StartSpeed = 200;
    params->c5SpeedSteps = 0;
    params->c5SkillDelayMin = 0;
    params->c5SkillDelaySteps = 0;

    return params;
}
/**/
// N is the maximum number of structs to insert
#define N 10000
#define MAX_TURN_METER 1428.57

typedef struct {
    int A, B, C;
} Match;

__device__ char dev_data[N];
__device__ int dev_count = 0;
__device__ int dev_founded = 0;

__device__ int my_push_back(char mt) {
    int insert_pt = atomicAdd(&dev_count, 1);
    if (insert_pt < N) {
        dev_data[insert_pt] = mt;
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

cudaError_t testWithCuda(Simulation* simulation, SimulationParams* params);

__device__ void tickTurnmeter(Entity* e) {
    // TODO other speed bufs / debufs
    if (e->speed30Duration > 0) {
        e->turnMeter += e->speed * 1.3;
    }
    else {
        e->turnMeter += e->speed;
    }
}

__device__ void tickAllTurnmeters(Simulation* s) {
    tickTurnmeter(&s->c1);
    tickTurnmeter(&s->c2);
    tickTurnmeter(&s->c3);
    tickTurnmeter(&s->c4);
    tickTurnmeter(&s->c5);
    tickTurnmeter(&s->cb);
}

__device__ bool makeClanBossTurn(Simulation* s) {
    s->cb.turnesMade++;
    s->cb.turnMeter = 0;
    my_push_back('B');

    if (s->cb.turnesMade > 3 && (s->cb.turnesMade % 3 == 1 || s->cb.turnesMade % 3 == 2)) {
        // starting checks on 4th turn. only when it is AOE 1-2
        return s->c1.unkillableDuration > 0 &&
            s->c2.unkillableDuration > 0 &&
            s->c3.unkillableDuration > 0 &&
            s->c4.unkillableDuration > 0 &&
            s->c5.unkillableDuration > 0;
    }

    return true;
}

__device__ void makeChampionTurn(Simulation* s, ChampionStruct* c) {
    c->turnMeter = 0;
    c->unkillableDuration--;
    c->speed30Duration--;
    // TODO other buffs

    // TMP log move
    switch (c->type)
    {
    case ApothecaryType:
        my_push_back('A');
        break;
    case WarcasterType:
        my_push_back('W');
        break;
    case KymerType:
        my_push_back('K');
        break;
    default:
        my_push_back('-');
    }

    if (c->skillDelay <= 0 && c->skillCurrentCooldown <= 0) {
        // perform skill
        switch (c->type)
        {
        case ApothecaryType:
            my_push_back('S');
            // Fills the Turn Meter of all allies by 15 %.
            s->c1.turnMeter += MAX_TURN_METER * 15 / 100;
            s->c2.turnMeter += MAX_TURN_METER * 15 / 100;
            s->c3.turnMeter += MAX_TURN_METER * 15 / 100;
            s->c4.turnMeter += MAX_TURN_METER * 15 / 100;
            s->c5.turnMeter += MAX_TURN_METER * 15 / 100;
            // Places a 30 % Increase Speed buff on all allies for 2 turns
            s->c1.speed30Duration = 2;
            s->c2.speed30Duration = 2;
            s->c3.speed30Duration = 2;
            s->c4.speed30Duration = 2;
            s->c5.speed30Duration = 2;
            break;
        case WarcasterType:
            my_push_back('S');
            //  Places block damage on all allies
            s->c1.unkillableDuration = 1;
            s->c2.unkillableDuration = 1;
            s->c3.unkillableDuration = 1;
            s->c4.unkillableDuration = 1;
            s->c5.unkillableDuration = 1;
            break;
        case KymerType:
            my_push_back('S');
            // Fills the Turn Meter of all allies by 20 %.

            if (&s->c1 != c) { s->c1.turnMeter += MAX_TURN_METER * 20 / 100; }
            if(&s->c2 != c) { s->c2.turnMeter += MAX_TURN_METER * 20 / 100; }
            if(&s->c3 != c) { s->c3.turnMeter += MAX_TURN_METER * 20 / 100; }
            if(&s->c4 != c) { s->c4.turnMeter += MAX_TURN_METER * 20 / 100; }
            if(&s->c5 != c) { s->c5.turnMeter += MAX_TURN_METER * 20 / 100; }


            // Resets the cooldown of ally skills
            // since we will put our skill on cooldown later, it's ok to reset our skill as well
            s->c1.skillCurrentCooldown = 0;
            s->c2.skillCurrentCooldown = 0;
            s->c3.skillCurrentCooldown = 0;
            s->c4.skillCurrentCooldown = 0;
            s->c5.skillCurrentCooldown = 0;


            break;
        default:
            break;
        }

        // put the skill on cooldown
        c->skillCurrentCooldown = c->skillCooldown;
    }

    c->skillCurrentCooldown--;
    c->skillDelay--;
}


__device__ bool makeTurn(Simulation* s) {

    if (s->cb.turnMeter >= MAX_TURN_METER ||
        s->c1.turnMeter >= MAX_TURN_METER ||
        s->c2.turnMeter >= MAX_TURN_METER ||
        s->c3.turnMeter >= MAX_TURN_METER ||
        s->c4.turnMeter >= MAX_TURN_METER ||
        s->c5.turnMeter >= MAX_TURN_METER)
    {
        if (s->c1.turnMeter >= s->cb.turnMeter &&
            s->c1.turnMeter >= s->c2.turnMeter &&
            s->c1.turnMeter >= s->c3.turnMeter &&
            s->c1.turnMeter >= s->c4.turnMeter &&
            s->c1.turnMeter >= s->c5.turnMeter)
        {
            makeChampionTurn(s, &s->c1);
        }
        else if (s->c2.turnMeter >= s->cb.turnMeter &&
            s->c2.turnMeter >= s->c1.turnMeter &&
            s->c2.turnMeter >= s->c3.turnMeter &&
            s->c2.turnMeter >= s->c4.turnMeter &&
            s->c2.turnMeter >= s->c5.turnMeter)
        {
            makeChampionTurn(s, &s->c2);
        }
        else if (s->c3.turnMeter >= s->cb.turnMeter &&
            s->c3.turnMeter >= s->c1.turnMeter &&
            s->c3.turnMeter >= s->c2.turnMeter &&
            s->c3.turnMeter >= s->c4.turnMeter &&
            s->c3.turnMeter >= s->c5.turnMeter)
        {
            makeChampionTurn(s, &s->c3);
        }
        else if (s->c4.turnMeter >= s->cb.turnMeter &&
            s->c4.turnMeter >= s->c1.turnMeter &&
            s->c4.turnMeter >= s->c2.turnMeter &&
            s->c4.turnMeter >= s->c3.turnMeter &&
            s->c4.turnMeter >= s->c5.turnMeter)
        {
            makeChampionTurn(s, &s->c4);
        }
        else if (s->c5.turnMeter >= s->cb.turnMeter &&
            s->c5.turnMeter >= s->c1.turnMeter &&
            s->c5.turnMeter >= s->c2.turnMeter &&
            s->c5.turnMeter >= s->c3.turnMeter &&
            s->c5.turnMeter >= s->c4.turnMeter)
        {
            makeChampionTurn(s, &s->c5);
        }
        else if (s->cb.turnMeter >= s->c1.turnMeter &&
            s->cb.turnMeter >= s->c2.turnMeter &&
            s->cb.turnMeter >= s->c3.turnMeter &&
            s->cb.turnMeter >= s->c4.turnMeter &&
            s->cb.turnMeter >= s->c5.turnMeter)
        {
            return makeClanBossTurn(s);
        }
    }
    return true;

}

__global__ void test(Simulation* simulation, SimulationParams* params) {
    //int i = threadIdx.x;
    //

    Simulation s;

    memcpy(&s, simulation, sizeof(Simulation));

    // initialize speeds
    s.c1.speed = 302;// params->c1StartSpeed;
    s.c2.speed = 286; // params->c2StartSpeed;
    s.c3.speed = 286; // params->c3StartSpeed;
    s.c4.speed = 270; // params->c4StartSpeed;
    s.c5.speed = 200; // params->c5StartSpeed;
    s.cb.speed = 190;// params->cbSpeed;

    // TODO initialize cooldowns
    s.c1.skillDelay = 1; // params->c1SkillDelayMin;
    s.c2.skillDelay = 0;// params->c2SkillDelayMin;
    s.c3.skillDelay = 2;// params->c3SkillDelayMin;
    s.c4.skillDelay = 0; // params->c4SkillDelayMin;
    s.c5.skillDelay = 0;// params->c5SkillDelayMin;
    // TODO get block/thread id and use steps during initialization
    bool running = true;
    
    while (running) {
        tickAllTurnmeters(&s);
        running = makeTurn(&s);
        if (s.cb.turnesMade >= 50) {
            // STOP after 50 turns
            running = false;
        }
    }
    /**/

    if (s.cb.turnesMade >= 50) {
        // SUCCESS
       atomicAdd(&dev_founded, 1);
    }


  //  if (i == 1) {
   //     Match* m = new Match;
    //    m->A = threadIdx.x;
    //    my_push_back(m);
    //}
    /*
   // result[0] = threadIdx.x;
   // result[1] = blockIdx.x;
    result[2] = threadIdx.z;
    result[3] = blockDim.x;
    result[4] = blockDim.z;*/
   // tickTurnmeter(c1);
   // return c1->speed;
//    delete s;
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
    cudaError_t cudaStatus = testWithCuda(x, params);
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
cudaError_t testWithCuda(Simulation* simulation, SimulationParams* params)
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
    int block_size = 1;
    int blocks_count = 1; // variations / block_size;
    //int blocks_count = 1;
    fprintf(stderr, "CalculateSimulationParamsVariations %u\nblock_size=%i\nblocks_count=%i\n", variations, block_size, blocks_count);


   



    // Launch a kernel on the GPU with one thread for each element.
    test <<<blocks_count, block_size >>>(gpuSimulation, gpuSimulationParams);

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

    int founded;
    cudaStatus = cudaMemcpyFromSymbol(&founded, dev_founded, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    printf("founded=%d\n", founded);

    int dsize;
    cudaStatus = cudaMemcpyFromSymbol(&dsize, dev_count, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    printf("gpuResult.count=%d\n", dsize);


    char result[N];
    cudaStatus = cudaMemcpyFromSymbol(&result, dev_data, N * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA cudaMemcpyFromSymbol result failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    for (int i = 0; i < 500; i++) {
        if (result[i] == 'S') {
            printf("S");
        }
        else if (result[i] == 'B') {
            printf("\tBOSS\n");
        }
        else {
            printf(" %c", result[i]);
        }
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
