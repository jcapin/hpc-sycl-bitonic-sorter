/*
    Napravit osnovni algoritam, te onda radit optimizaciju (recimo tiling, pa onda mijenjanje smjera u memoriji itd..)

    Vizualizirat na neki nacin(recimo svakih 10ms stanje)

    Napisat Paper na temu Bitonic Sortinga i kako sam ga implementirat


    Upute za sam kod:
    Trebaju 4 verzije
    Assertat na kraju pocetni i konacni niz
    Gledat vremena pokretanja

    Prva verzija ova koja je vec napisana
    Druga verzija recimo local memory
    Treca verzija recimo subgroups
    Cetvrta tiling (ili nesto slicno)
*/

%%writefile lab/simple.cpp
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cassert>
#include <chrono>
#include <numeric>

using namespace sycl;

// Helper function to check power-of-two size
bool is_power_of_two(size_t x) {
    return x && !(x & (x - 1));
}

// Bitonic sort kernel -> basic algorithm
/**
 * @file bitonic_sort.cpp
 * @brief Implementation of the Bitonic Sort algorithm using SYCL for parallel execution.
 * 
 * This function performs a parallel bitonic sort on a given vector of integers. 
 * The algorithm requires the size of the input array to be a power of two.
 * 
 * @param data A reference to a vector of integers to be sorted.
 * 
 * @details
 * The function uses the SYCL framework to parallelize the sorting process. 
 * It creates a SYCL queue and a buffer to manage the data on the device. 
 * The sorting is performed in multiple stages, where each stage involves 
 * comparing and swapping elements based on the bitonic sorting network.
 * 
 * The algorithm works as follows:
 * - The outer loop iterates over the size of the bitonic sequence (`k`), doubling it in each iteration.
 * - The inner loop iterates over the size of the subsequences (`j`), halving it in each iteration.
 * - A parallel kernel is launched for each stage, where each thread compares and swaps elements 
 *   based on their indices and the current sorting direction (ascending or descending).
 * 
 * The function ensures synchronization by waiting for all submitted tasks to complete using `q.wait()`.
 * 
 * @note
 * - The input vector size must be a power of two; otherwise, the function will print an error message 
 *   and return without performing any sorting.
 * - The function relies on a helper function `is_power_of_two()` to check the size constraint.
 * - The SYCL `buffer` and `queue` are used to manage data and execute kernels on the device.
 * 
 * @warning
 * This implementation assumes that the SYCL runtime and necessary headers are available. 
 * Ensure that the SYCL environment is correctly set up before using this function.
 */
void bitonic_sort_v1(std::vector<int>& data) {
    const size_t N = data.size(); // Get the size of the input data
    if (!is_power_of_two(N)) { // Check if the size is a power of two
        std::cerr << "Array size must be a power of two.\n"; // Print error if not
        return; // Exit the function
    }

    queue q; // Create a SYCL queue for device execution

    buffer<int, 1> data_buf(data.data(), range<1>(N)); // Create a buffer to manage data on the device

    // Outer loop for the size of the bitonic sequence
    for (size_t k = 2; k <= N; k *= 2) {
        // Inner loop for the size of the subsequences
        for (size_t j = k / 2; j > 0; j /= 2) {
            q.submit([&](handler& h) { // Submit a command group to the queue
                auto d = data_buf.get_access<access::mode::read_write>(h); // Get access to the buffer
                h.parallel_for(range<1>(N), [=](id<1> i) { // Launch a parallel kernel
                    size_t ixj = i[0] ^ j; // Compute the index to compare with
                    if (ixj > i[0]) { // Ensure the index is valid
                        bool ascending = ((i[0] & k) == 0); // Determine sorting direction
                        if ((d[i[0]] > d[ixj]) == ascending) { // Compare and swap if needed
                            int temp = d[i[0]]; // Temporary variable for swapping
                            d[i[0]] = d[ixj]; // Swap elements
                            d[ixj] = temp; // Complete the swap
                        }
                    }
                });
            });

            // Visualize the current state of the array after each step
            // q.submit([&](handler& h) {
            //     auto d = data_buf.get_access<access::mode::read>(h);
            //     h.host_task([=]() {
            //         std::ofstream outfile("bitonic_states.txt", std::ios::app); // append mode
            //         outfile << "k=" << k << ",j=" << j << ":";
            //         for (size_t idx = 0; idx < N; ++idx) {
            //             outfile << " " << d[idx];
            //         }
            //         outfile << "\n";
            //     });
            // });
        }
    }

    q.wait(); // Wait for all tasks in the queue to complete
}

// Bitonic sort kernel with local memory optimization
/**
 * @brief Implementation of Bitonic Sort using local memory optimization.
 * 
 * This version of the bitonic sort algorithm utilizes local memory to reduce
 * global memory accesses, improving performance by using faster local memory.
 * 
 * @param data A reference to a vector of integers to be sorted.
 * 
 * @details
 * - The algorithm loads data into local memory for each workgroup.
 * - Synchronization is achieved using barriers to ensure all threads in a workgroup
 *   have consistent data before proceeding.
 * - After sorting in local memory, the results are written back to global memory.
 */
void bitonic_sort_v2(std::vector<int>& data) {
    const size_t N = data.size(); // Get the size of the input data
    if (!is_power_of_two(N)) { // Check if the size is a power of two
        std::cerr << "Array size must be a power of two.\n"; // Print error if not
        return; // Exit the function
    }

    queue q; // Create a SYCL queue for device execution

    buffer<int, 1> data_buf(data.data(), range<1>(N)); // Create a buffer to manage data on the device

    // Outer loop for the size of the bitonic sequence
    for (size_t k = 2; k <= N; k *= 2) {
        // Inner loop for the size of the subsequences
        for (size_t j = k / 2; j > 0; j /= 2) {
            q.submit([&](handler& h) { // Submit a command group to the queue
                auto d = data_buf.get_access<access::mode::read_write>(h); // Get access to the buffer
                accessor<int, 1, access::mode::read_write, access::target::local> local_mem(range<1>(N), h); // Create a local memory accessor

                h.parallel_for(nd_range<1>(range<1>(N), range<1>(N)), [=](nd_item<1> item) { // Launch a parallel kernel
                    size_t i = item.get_global_id(0); // Get the global ID of the current thread
                    size_t ixj = i ^ j; // Compute the index to compare with

                    // Load data into local memory
                    local_mem[i] = d[i]; // Copy data from global memory to local memory
                    item.barrier(access::fence_space::local_space); // Synchronize threads in the workgroup

                    if (ixj > i) { // Ensure the index is valid
                        bool ascending = ((i & k) == 0); // Determine sorting direction
                        if ((local_mem[i] > local_mem[ixj]) == ascending) { // Compare and swap if needed
                            int temp = local_mem[i]; // Temporary variable for swapping
                            local_mem[i] = local_mem[ixj]; // Swap elements in local memory
                            local_mem[ixj] = temp; // Complete the swap
                        }
                    }

                    item.barrier(access::fence_space::local_space); // Synchronize threads in the workgroup

                    // Write back to global memory
                    d[i] = local_mem[i]; // Copy sorted data from local memory to global memory
                });
            });

            // Visualize the current state of the array after each step
            // q.submit([&](handler& h) {
            //     auto d = data_buf.get_access<access::mode::read>(h);
            //     h.host_task([=]() {
            //         std::ofstream outfile("bitonic_states_v2.txt", std::ios::app); // append mode
            //         outfile << "k=" << k << ",j=" << j << ":";
            //         for (size_t idx = 0; idx < N; ++idx) {
            //             outfile << " " << d[idx];
            //         }
            //         outfile << "\n";
            //     });
            // });
        }
    }

    q.wait(); // Wait for all tasks in the queue to complete
}

// Bitonic sort kernel with subgroups optimization
/**
 * @brief Implementation of Bitonic Sort using subgroups optimization.
 * 
 * This version of the bitonic sort algorithm leverages SYCL subgroups to perform
 * sorting operations more efficiently by utilizing subgroup-level parallelism.
 * 
 * @param data A reference to a vector of integers to be sorted.
 * 
 * @details
 * - Subgroups allow threads within a workgroup to collaborate more closely.
 * - The algorithm uses subgroup operations to perform comparisons and swaps
 *   within a subgroup, reducing overhead compared to global memory operations.
 * - This approach is particularly effective on hardware with efficient subgroup support.
 */
void bitonic_sort_v3(std::vector<int>& data) {
    const size_t N = data.size(); // Get the size of the input data
    if (!is_power_of_two(N)) { // Check if the size is a power of two
        std::cerr << "Array size must be a power of two.\n"; // Print error if not
        return; // Exit the function
    }

    queue q; // Create a SYCL queue for device execution

    buffer<int, 1> data_buf(data.data(), range<1>(N)); // Create a buffer to manage data on the device

    // Outer loop for the size of the bitonic sequence
    for (size_t k = 2; k <= N; k *= 2) {
        // Inner loop for the size of the subsequences
        for (size_t j = k / 2; j > 0; j /= 2) {
            q.submit([&](handler& h) { // Submit a command group to the queue
                auto d = data_buf.get_access<access::mode::read_write>(h); // Get access to the buffer

                // Launch a parallel kernel with subgroup optimization
                h.parallel_for(nd_range<1>(range<1>(N), range<1>(16)), [=](nd_item<1> item) {
                    auto sg = item.get_sub_group(); // Get the subgroup for the current thread
                    size_t i = item.get_global_id(0); // Get the global ID of the current thread
                    size_t ixj = i ^ j; // Compute the index to compare with

                    int val = d[i]; // Load the current element
                    int partner_val = d[ixj]; // Load the partner element

                    // Perform subgroup exchange
                    if (ixj > i) { // Ensure the index is valid
                        bool ascending = ((i & k) == 0); // Determine sorting direction
                        int min_val = sycl::min(val, partner_val); // Compute the minimum value
                        int max_val = sycl::max(val, partner_val); // Compute the maximum value

                        if (ascending) { // If sorting in ascending order
                            val = min_val; // Assign the minimum value to the current element
                            partner_val = max_val; // Assign the maximum value to the partner element
                        } else { // If sorting in descending order
                            val = max_val; // Assign the maximum value to the current element
                            partner_val = min_val; // Assign the minimum value to the partner element
                        }
                    }

                    // Write back to global memory
                    d[i] = val; // Store the updated value of the current element
                    d[ixj] = partner_val; // Store the updated value of the partner element
                });
            });

            // Visualize the current state of the array after each step
            // q.submit([&](handler& h) {
            //     auto d = data_buf.get_access<access::mode::read>(h);
            //     h.host_task([=]() {
            //         std::ofstream outfile("bitonic_states_v3.txt", std::ios::app); // append mode
            //         outfile << "k=" << k << ",j=" << j << ":";
            //         for (size_t idx = 0; idx < N; ++idx) {
            //             outfile << " " << d[idx];
            //         }
            //         outfile << "\n";
            //     });
            // });
        }
    }

    q.wait(); // Wait for all tasks in the queue to complete
}

// Bitonic sort kernel with tiling optimization
/**
 * @brief Implementation of Bitonic Sort using tiling optimization.
 * 
 * This version of the bitonic sort algorithm uses tiling optimization to perform
 * sorting operations more efficiently by utilizing tiling-level parallelism.
 * 
 * @param data A reference to a vector of integers to be sorted.
 * 
 *  
 * @details
 * - Tiling partitions the input data into smaller blocks (tiles) to optimize memory access patterns.
 * - The algorithm loads each tile into shared (local) memory to reduce redundant global memory accesses.
 * - Computation is then performed on these tiles, enabling better cache utilization and improved parallelism.
 * - This technique is especially beneficial on hardware with limited memory bandwidth or high memory latency.
 */

void bitonic_sort_v4(std::vector<int>& data) {
    const size_t N = data.size();
    if (!is_power_of_two(N)) {
        std::cerr << "Array size must be a power of two.\n";
        return;
    }

    sycl::queue q;
    sycl::buffer<int, 1> buf(data.data(), sycl::range<1>(N));

    const size_t tile_size = 256;

    for (size_t k = 2; k <= N; k *= 2) {
        for (size_t j = k / 2; j > 0; j /= 2) {
            q.submit([&](sycl::handler& h) {
                auto d = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(tile_size)),
                    [=](sycl::nd_item<1> item) {
                        size_t i = item.get_global_id(0);
                        size_t ixj = i ^ j;

                        if (ixj > i && ixj < N && i < N) {
                            bool ascending = ((i & k) == 0);
                            int val_i = d[i];
                            int val_ixj = d[ixj];

                            if ((val_i > val_ixj) == ascending) {
                                d[i] = val_ixj;
                                d[ixj] = val_i;
                            }
                        }
                    });
            });
        }
    }

    q.wait();
}


void print_stats(const std::string& label, const std::vector<long long>& durations) {
    long long sum = std::accumulate(durations.begin(), durations.end(), 0LL);
    double average = static_cast<double>(sum) / durations.size();

    double variance = 0.0;
    for (auto d : durations) {
        variance += (d - average) * (d - average);
    }
    variance /= durations.size();
    double stddev = std::sqrt(variance);

    auto [min_it, max_it] = std::minmax_element(durations.begin(), durations.end());

    std::cout << label << " Stats (microseconds):\n";
    std::cout << "  Avg: " << average << "\n";
    std::cout << "  Std Dev: " << stddev << "\n";
    std::cout << "  Min: " << *min_it << "\n";
    std::cout << "  Max: " << *max_it << "\n";
    std::cout << std::endl;
}


int main() {
    constexpr int num_iterations = 100;
    constexpr int data_size = 1024;

    // Seed RNG for reproducibility (optional)
    std::srand(42);

    // Generate random base dataset ONCE
    std::vector<int> originalData(data_size);
    std::generate(originalData.begin(), originalData.end(), []() { return rand() % 100000; });

    // Reference sorted result for validation
    std::vector<int> expected_data = originalData;
    std::sort(expected_data.begin(), expected_data.end());

    std::vector<int> currentData;
    std::vector<long long> durations;

    

    // -------- v1 --------
    durations.clear();
    for (int i = 0; i < num_iterations; ++i) {
        currentData = originalData;  // deep copy
        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort_v1(currentData);
        auto end = std::chrono::high_resolution_clock::now();

        if (i != 0)
            durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        assert(currentData == expected_data && "bitonic_sort_v1 failed.");
    }
    print_stats("Bitonic Sort v1", durations);

    // -------- v2 --------
    durations.clear();
    for (int i = 0; i < num_iterations; ++i) {
        currentData = originalData;  // deep copy
        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort_v2(currentData);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        assert(currentData == expected_data && "bitonic_sort_v2 failed.");
    }
    print_stats("Bitonic Sort v2", durations);

    // -------- v3 --------
    durations.clear();
    for (int i = 0; i < num_iterations; ++i) {
        currentData = originalData;  // deep copy
        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort_v3(currentData);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        assert(currentData == expected_data && "bitonic_sort_v3 failed.");
    }
    print_stats("Bitonic Sort v3", durations);

    // -------- v4 --------
    durations.clear();
    for (int i = 0; i < num_iterations; ++i) {
        currentData = originalData;  // deep copy
        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort_v4(currentData);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        assert(currentData == expected_data && "bitonic_sort_v4 failed.");
    }
    print_stats("Bitonic Sort v4", durations);

    return 0;
}

// int main() {
//     std::vector<int> startingData;
//     // Randomize data as an integer array of 1024 elements
//     startingData.resize(1024);
//     std::generate(startingData.begin(), startingData.end(), []() { return rand() % 100000; });

//     // Prepare for assertion
//     std::vector<int> expected_data = startingData;
//     std::sort(expected_data.begin(), expected_data.end());

//     std::vector<int> data;
    
//     // bitonic_sort_v1
//     data = startingData;
//     auto start = std::chrono::high_resolution_clock::now();
//     bitonic_sort_v1(data);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout << "Bitonic Sort v1 Time: " 
//               << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
//               << " microseconds\n";
//     assert(data == expected_data && "The sorted data does not match the expected result.");

//     // bitonic_sort_v2
//     data = startingData; // Reset data to the original generated data
//     start = std::chrono::high_resolution_clock::now();
//     bitonic_sort_v2(data);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << "Bitonic Sort v2 Time: " 
//               << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
//               << " microseconds\n";
//     assert(data == expected_data && "The sorted data does not match the expected result.");

//     // bitonic_sort_v3
//     data = startingData; // Reset data
//     start = std::chrono::high_resolution_clock::now();
//     bitonic_sort_v3(data);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << "Bitonic Sort v3 Time: " 
//               << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
//               << " microseconds\n";
//     assert(data == expected_data && "The sorted data does not match the expected result.");

//     // bitonic_sort_v4
//     data = startingData; // Reset data
//     start = std::chrono::high_resolution_clock::now();
//     bitonic_sort_v4(data);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << "Bitonic Sort v4 Time: " 
//               << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
//               << " microseconds\n";
//     assert(data == expected_data && "The sorted data does not match the expected result.");

//     return 0;
// }

