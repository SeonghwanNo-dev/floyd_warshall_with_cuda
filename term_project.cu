#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility> // std::pair 사용
#include <cstring> // memcpy 사용
#include <cuda_runtime.h>

using namespace std;
const float INF = 1e9f;
const int BLOCK_SIZE = 32;

// Functions
pair<float*, float*> floyd_space_generate_dual(int n);
void floyd_exe_CPU(int n, float* graph);
void floyd_setup_and_exe_GPU(int n, float* graph, const char* version);

// CUDA Kernel
__global__ void floyd_exe_GPU_base(float* graph, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= n && j <= n) {
        int idx_ij = i * (n + 1) + j;
        int idx_ik = i * (n + 1) + k;
        int idx_kj = k * (n + 1) + j;

        if (graph[idx_ik] != INF && graph[idx_kj] != INF) {
            if (graph[idx_ij] > graph[idx_ik] + graph[idx_kj]) {
                graph[idx_ij] = graph[idx_ik] + graph[idx_kj];
            }
        }
    }
}

__global__ void floyd_exe_GPU_v1(float* graph, int n, int k) {
}

// Main
int main() {
    for (int n = 35000; n <= 40000; n += 1000) {
        auto [graph_cpu, graph_gpu] = floyd_space_generate_dual(n);

        cout << "--- CPU 수행 시작 ---" << endl;
        floyd_exe_CPU(n, graph_cpu);

        cout << "--- GPU_base 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_gpu, "base");

        // cout << "--- GPU_v1 수행 시작 ---" << endl;
        // floyd_setup_and_exe_GPU(n, graph_gpu, "v1");

        delete[] graph_cpu;
        delete[] graph_gpu;
    }
    return 0;
}

// 두 개의 그래프 배열을 한 번에 생성하여 반환하는 함수
pair<float*, float*> floyd_space_generate_dual(int n) {
    random_device rd;
    mt19937 gen(rd());

    int m = n * 3;
    long long array_size = (long long)(n + 1) * (n + 1);

    cout << "========================================" << endl;
    cout << "[테스트 환경] 노드 수(n): " << n << ", 간선 수(m): " << m << endl;

    float* g1 = new float[array_size];
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            g1[i * (n + 1) + j] = INF;
        }
        g1[i * (n + 1) + i] = 0.0f;
    }

    // 간선 생성
    uniform_int_distribution<int> dis(1, n);
    int edges_generated = 0;
    while (edges_generated < m) {
        int a = dis(gen);
        int b = dis(gen);
        long long idx_ab = (long long)a * (n + 1) + b;
        long long idx_ba = (long long)b * (n + 1) + a;

        if (a != b && g1[idx_ab] == INF) {
            g1[idx_ab] = 1.0f;
            g1[idx_ba] = 1.0f;
            edges_generated++;
        }
    }

    float* g2 = new float[array_size];
    memcpy(g2, g1, array_size * sizeof(float));

    return {g1, g2};
}

void floyd_exe_CPU(int n, float* graph){
    auto start = chrono::high_resolution_clock::now();
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                int idx_ij = i * (n + 1) + j;
                int idx_ik = i * (n + 1) + k;
                int idx_kj = k * (n + 1) + j;

                if (graph[idx_ik] != INF && graph[idx_kj] != INF) {
                    if (graph[idx_ij] > graph[idx_ik] + graph[idx_kj]) {
                        graph[idx_ij] = graph[idx_ik] + graph[idx_kj];
                    }
                }
            }
        }
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    double total_sum = 0;
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            if(graph[i * (n + 1) + j] != INF) total_sum += graph[i * (n + 1) + j];
        }
    }

    cout << "-> CPU 알고리즘 수행 시간: " << duration.count() << " ms" << endl;
    cout << "-> 결과 (유효 경로 가중치 총합): " << total_sum << endl;
    cout << "========================================\n" << endl;
}

void floyd_setup_and_exe_GPU(int n, float* graph, const char* version){
    auto start = chrono::high_resolution_clock::now();

    cudaSetDevice(0);

    long long array_size = (long long)(n + 1) * (n + 1);
    float *dev_graph;

    cudaMalloc((void **)&dev_graph, array_size * sizeof(float));
    cudaMemcpy(dev_graph, graph, array_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (strcmp(version, "base") == 0) {
        for (int k = 1; k <= n; k++) {
            floyd_exe_GPU_base<<<dimGrid, dimBlock>>>(dev_graph, n, k);
        }
    }
    else if (strcmp(version, "v1") == 0) {
        for (int k = 1; k <= n; k++) {
            floyd_exe_GPU_v1<<<dimGrid, dimBlock>>>(dev_graph, n, k);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(graph, dev_graph, array_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_graph);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    double total_sum = 0;
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            if(graph[i * (n + 1) + j] != INF) total_sum += graph[i * (n + 1) + j];
        }
    }
    cout << "-> GPU 알고리즘 수행 시간: " << duration.count() << " ms" << endl;
    cout << "-> 결과 (유효 경로 가중치 총합): " << total_sum << endl;
    cout << "========================================\n" << endl;
}
