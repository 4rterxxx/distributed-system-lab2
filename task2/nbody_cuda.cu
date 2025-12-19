#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define G 6.67430e-11f
#define DT 3600.0f
#define EPS 1e-6f
#define BLOCK_SIZE 256

typedef struct {
    float mass;
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
} Particle;

/* ===== CUDA kernels ===== */

__global__ void compute_accelerations(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    float xi = p[i].x;
    float yi = p[i].y;
    float zi = p[i].z;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = p[j].x - xi;
        float dy = p[j].y - yi;
        float dz = p[j].z - zi;

        float dist2 = dx * dx + dy * dy + dz * dz + EPS;
        float invDist = rsqrtf(dist2);
        float invDist3 = invDist * invDist * invDist;

        float factor = G * p[j].mass * invDist3;
        axi += factor * dx;
        ayi += factor * dy;
        azi += factor * dz;
    }

    p[i].ax = axi;
    p[i].ay = ayi;
    p[i].az = azi;
}

__global__ void update_particles(Particle* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    p[i].vx += p[i].ax * dt;
    p[i].vy += p[i].ay * dt;
    p[i].vz += p[i].az * dt;

    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

/* ===== Host code ===== */

int read_particles(const char* filename, Particle** p, int* n) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;

    fscanf(file, "%d", n);
    *p = (Particle*)malloc((*n) * sizeof(Particle));

    for (int i = 0; i < *n; i++) {
        fscanf(file, "%f %f %f %f %f %f %f",
               &(*p)[i].mass,
               &(*p)[i].x, &(*p)[i].y, &(*p)[i].z,
               &(*p)[i].vx, &(*p)[i].vy, &(*p)[i].vz);
        (*p)[i].ax = (*p)[i].ay = (*p)[i].az = 0.0f;
    }

    fclose(file);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <tend> <input_file> <output_file> <blocks>\n", argv[0]);
        return 1;
    }

    float tend = atof(argv[1]);
    char* input_file = argv[2];
    char* output_file = argv[3];
    int blocks = atoi(argv[4]);

    int n;
    Particle* h_particles;

    if (read_particles(input_file, &h_particles, &n) != 0) {
        printf("Error reading input file\n");
        return 1;
    }

    Particle* d_particles;
    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, n * sizeof(Particle),
               cudaMemcpyHostToDevice);

    FILE* out = fopen(output_file, "w");
    if (!out) {
        printf("Error opening output file\n");
        return 1;
    }

    int steps = (int)(tend / DT);

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {
        compute_accelerations<<<grid, block>>>(d_particles, n);
        update_particles<<<grid, block>>>(d_particles, DT, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float exec_time_ms = 0.0f;
    cudaEventElapsedTime(&exec_time_ms, start, stop);

    printf("Particles: %d\n", n);
    printf("Steps: %d\n", steps);
    printf("Execution time (CUDA): %.6f ms\n", exec_time_ms);
    printf("Average time per step: %.6f ms\n", exec_time_ms / steps);

    fclose(out);
    cudaFree(d_particles);
    free(h_particles);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
