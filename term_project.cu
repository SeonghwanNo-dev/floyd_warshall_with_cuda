#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility> // std::pair 사용
#include <cstring> // memcpy 사용
#include <string>
#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>


using namespace std;
const float INF = 1e9f;
const int BLOCK_SIZE = 32;

// Functions
pair<float*, float*> floyd_space_generate_dual(int n);
void floyd_exe_CPU(int n, float* graph);
void save_log_to_json(const std::string& version, double elapsed_time, long long total_weight);
void floyd_setup_and_exe_GPU(int n, float* graph, const char* version);


// CUDA Kernel (Base)
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

// CUDA Kernel (Fully Optimized)
__global__ void floyd_exe_GPU_v1(float* graph, int n, int k) {
    // All Optimization Strategies Implemented
    // Planned for ablation study
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // 1. Shared Memory Usage
    __shared__ float sh_ik[BLOCK_SIZE];
    __shared__ float sh_kj[BLOCK_SIZE];

    int idx_ij = i * (n + 1) + j;
    int idx_ik = i * (n + 1) + k;
    int idx_kj = k * (n + 1) + j;

    if (tx == 0 && i <= n) sh_ik[ty] = graph[idx_ik];
    if (ty == 0 && j <= n) sh_kj[tx] = graph[idx_kj];
    
    // 2. Register Usage
    float current_dist = 0.0f;
    if (i <= n && j <= n) current_dist = graph[idx_ij];
    __syncthreads();

    if (i <= n && j <= n) {
        if (sh_ik[ty] != INF && sh_kj[tx] != INF) {
            // 3. Branchless update using fminf
            current_dist = fminf(current_dist, sh_ik[ty] + sh_kj[tx]);
            graph[idx_ij] = current_dist;
        }
    }
}

// CUDA Kernel (Except Shared Memory Usage)
__global__ void floyd_exe_GPU_v2(float* graph, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= n && j <= n) {
        int idx_ij = i * (n + 1) + j;
        int idx_ik = i * (n + 1) + k;
        int idx_kj = k * (n + 1) + j;

        // 2. Register Usage
        float current_dist = graph[idx_ij];

        if (graph[idx_ik] != INF && graph[idx_kj] != INF) {
            // 3. Branchless update using fminf
            current_dist = fminf(current_dist, graph[idx_ik] + graph[idx_kj]);
            graph[idx_ij] = current_dist;
        }
    }
}


// CUDA Kernel (Except Register Usage)
__global__ void floyd_exe_GPU_v3(float* graph, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // 1. Shared Memory Usage
    __shared__ float sh_ik[BLOCK_SIZE];
    __shared__ float sh_kj[BLOCK_SIZE];

    int idx_ij = i * (n + 1) + j;
    int idx_ik = i * (n + 1) + k;
    int idx_kj = k * (n + 1) + j;

    if (tx == 0 && i <= n) sh_ik[ty] = graph[idx_ik];
    if (ty == 0 && j <= n) sh_kj[tx] = graph[idx_kj];
    __syncthreads();

    if (i <= n && j <= n) {
        if (sh_ik[ty] != INF && sh_kj[tx] != INF) {
            // 3. Branchless update using fminf
            graph[idx_ij] = fminf(graph[idx_ij], sh_ik[ty] + sh_kj[tx]);
        }
    }
}



// CUDA Kernel (Except fminf)
__global__ void floyd_exe_GPU_v4(float* graph, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // 1. Shared Memory Usage
    __shared__ float sh_ik[BLOCK_SIZE];
    __shared__ float sh_kj[BLOCK_SIZE];

    int idx_ij = i * (n + 1) + j;
    int idx_ik = i * (n + 1) + k;
    int idx_kj = k * (n + 1) + j;

    if (tx == 0 && i <= n) sh_ik[ty] = graph[idx_ik];
    if (ty == 0 && j <= n) sh_kj[tx] = graph[idx_kj];

    // 2. Register Usage
    float current_dist = 0.0f;
    if (i <= n && j <= n) current_dist = graph[idx_ij];
    __syncthreads();

    if (i <= n && j <= n) {
        if (sh_ik[ty] != INF && sh_kj[tx] != INF) {
            if (current_dist > sh_ik[ty] + sh_kj[tx]) {
                current_dist = sh_ik[ty] + sh_kj[tx];
                graph[idx_ij] = current_dist;
            }
        }
    }
}



// Main
int main() {
    for (int n = 35000; n <= 40000; n += 500) {
        auto [graph_1, graph_2] = floyd_space_generate_dual(n);
        long long array_size = (long long)(n + 1) * (n + 1);


        cout << "--- CPU 수행 시작 ---" << endl;
        floyd_exe_CPU(n, graph_2);
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        cout << "--- GPU_base 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_2, "base");
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        cout << "--- GPU_v1 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_2, "v1");
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        cout << "--- GPU_v2 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_2, "v2");
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        cout << "--- GPU_v3 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_2, "v3");
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        cout << "--- GPU_v4 수행 시작 ---" << endl;
        floyd_setup_and_exe_GPU(n, graph_2, "v4");
        memcpy(graph_2, graph_1, array_size * sizeof(float));

        delete[] graph_1;
        delete[] graph_2;
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

// JSON 로그를 파일에 누적 저장하는 함수
void save_log_to_json(int n, const std::string& version, double elapsed_time, long long total_weight) {
    const std::string file_path = "./result.txt";
    
    std::ofstream log_file(file_path, std::ios::app);
    
    if (log_file.is_open()) {
        log_file << "{"
                 << "\"n\": " << n << ", "
                 << "\"version\": \"" << version << "\", "
                 << "\"elapsed_time_ms\": " << std::fixed << std::setprecision(6) << elapsed_time << ", "
                 << "\"total_weight\": " << total_weight
                 << "}\n";
                 
        log_file.close();
    } else {
        std::cerr << "[오류] " << file_path << " 파일을 열 수 없습니다." << std::endl;
    }
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
    else if (strcmp(version, "v2") == 0) {
        for (int k = 1; k <= n; k++) {
            floyd_exe_GPU_v2<<<dimGrid, dimBlock>>>(dev_graph, n, k);
        }
    }
    else if (strcmp(version, "v3") == 0) {
        for (int k = 1; k <= n; k++) {
            floyd_exe_GPU_v3<<<dimGrid, dimBlock>>>(dev_graph, n, k);
        }
    }
    else if (strcmp(version, "v4") == 0) {
        for (int k = 1; k <= n; k++) {
            floyd_exe_GPU_v4<<<dimGrid, dimBlock>>>(dev_graph, n, k);
        }
    }
    else {
        fprintf(stderr, "Error: Invalid version '%s'. Please choose from [base, v1, v2, v3, v4].\n", version);
        exit(1);
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
    save_log_to_json(n, version, duration.count(), total_sum);
    cout << "========================================\n" << endl;
}

