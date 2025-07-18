% % writefile lab / simple.cpp
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>

					using namespace sycl;

bool is_power_of_two(size_t x) { return x && !(x & (x - 1)); }

// Basic bitonic sort using SYCL with global memory only
void bitonic_sort_v1(std::vector<int> &data)
{
	size_t N = data.size();

	// Ensure the input size is a power of two
	assert(is_power_of_two(N) && "size must be power of two");

	// Create SYCL queue and buffer
	queue q;
	buffer<int> buf(data.data(), range<1>(N));

	// Outer loop: controls the size of bitonic sequences
	for (size_t k = 2; k <= N; k <<= 1)
	{
		// Inner loop: controls comparison distance
		for (size_t j = k >> 1; j > 0; j >>= 1)
		{
			q.submit([&](handler &h)
					 {
                        // Get read-write access to buffer
                        auto d = buf.get_access<access::mode::read_write>(h);

                        // Launch kernel with 1 thread per element
                        h.parallel_for(range<1>(N), [=](id<1> idx) {
                            size_t i = idx[0];
                            size_t ixj = i ^ j; // Bitonic compare index

                            // Only compare and possibly swap if ixjj > i
                            if (ixj > i && ((d[i] > d[ixj]) == ((i & k) == 0))) {
                                // Swap elements if out of order
                                int tmp = d[i];
                                d[i] = d[ixj];
                                d[ixj] = tmp;
                            }
                        }); });
		}
	}

	// Wait for all kernels to finish
	q.wait();
}

// Bitonic sort using local memory — fast, but limited to small N (≤ 8192)
void bitonic_sort_v2(std::vector<int> &data)
{
	const size_t N = data.size();

	// Check if input size is valid (power of two)
	if (!is_power_of_two(N))
	{
		std::cerr << "Array size must be a power of two.\n";
		return;
	}

	// Limit to fit entirely in local memory (per work-group)
	if (N > 8192)
	{
		std::cerr << "bitonic_sort_v2 supports only N <= 8192 due to local memory limits.\n";
		return;
	}

	// Create SYCL queue and buffer
	queue q;
	buffer<int> buf(data.data(), range<1>(N));

	// Submit one kernel that does the entire bitonic sort using local memory
	q.submit([&](handler &h)
			 {
		// Alllocate shared local memory for the work-group
		accessor<int, 1, access::mode::read_write, access::target::local> local_mem(N, h);
		auto d = buf.get_access<access::mode::read_write>(h);

		// Launch 1 work-group with N work-items (assumes full occupancy)
		h.parallel_for(nd_range<1>(range<1>(N), range<1>(N)), [=](nd_item<1> item) {
				size_t i = item.get_local_id(0); // Local ID within the group

				// Load input from global to local memory
				local_mem[i] = d[i];
				item.barrier(access::fence_space::local_space);

				// Perform full bitonic sort entirely within local memory
				for (size_t k = 2; k <= N; k <<= 1) {
						for (size_t j = k >> 1; j > 0; j >>= 1) {
								size_t ixj = i ^ j;
								if (ixj > i) {
										bool ascending = ((i & k) == 0);
										int a = local_mem[i];
										int b = local_mem[ixj];

										// Compare and swap if needed
										if ((a > b) == ascending) {
												local_mem[i] = b;
												local_mem[ixj] = a;
										}
								}
								// Synchronize after each compare-swap step
								item.barrier(access::fence_space::local_space);
						}
				}

				// Write sorted result back to global memory
				d[i] = local_mem[i];
		}); });
	q.wait();
}

// Performs bitonic sort using SYCL and parallel execution with subgroups (without explicit shuffle)
void bitonic_sort_v3(std::vector<int> &data)
{
	const size_t N = data.size();

	// Ensure input size is a power of two (bitonic sort requirement)
	assert(is_power_of_two(N));

	// Create SYCL queue for submitting kernels
	sycl::queue q;

	// Create a SYCL buffer from input data
	sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

	// Set local workkgroup size (tunable, depending on device)
	const size_t local_size = 128;

	// Outer loop: length of bitonic sequence to be merged
	for (size_t k = 2; k <= N; k <<= 1)
	{
		// Inner loop: distance between compared elements
		for (size_t j = k >> 1; j > 0; j >>= 1)
		{
			// Submit kernel for current (k, j) stage
			q.submit([&](sycl::handler &h)
					 {
			// Get read-write access to the buffer
			auto d = buf.get_access<sycl::access::mode::read_write>(h);

			// Launch parallel kernel across all elements
			h.parallel_for(
					sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(local_size)),
					[=](sycl::nd_item<1> item) {
							// Global index of the current item
							size_t i = item.get_global_id(0);

							// Compute index of the element to compare with
							size_t ixj = i ^ j;

							// Compare-and-swap if ixj > i (to avoid double swap) and both are within bounds
							if (ixj > i && ixj < N) {
									int val_i = d[i];
									int val_ixj = d[ixj];

									// Determine sorting direction for this stage
									bool ascending = ((i & k) == 0);

									// Swap if elements are in the wrong order
									if ((val_i > val_ixj) == ascending) {
											d[i] = val_ixj;
											d[ixj] = val_i;
									}
							}
					}); });
			q.wait(); // Wait for krnel to finish before continuing to next stage
		}
	}
}

// Performs bitonic sort using SYCL with tiling (workgroup-based parallel execution)
void bitonic_sort_v4(std::vector<int> &data)
{
	const size_t N = data.size();

	// Ensure input size is a power of two (requirement of bitonic sort)
	if (!is_power_of_two(N))
	{
		std::cerr << "Array size must be a power of two.\n";
		return;
	}

	// Create a SYCL queue for kernel submission
	sycl::queue q;

	// Create a SYCL buffer from input data
	sycl::buffer<int, 1> buf(data.data(), sycl::range<1>(N));

	// Set tile (workgroup) size – can be tuned depending on hardware
	const size_t tile_size = 256;

	// Outer loop: sequence size being merged
	for (size_t k = 2; k <= N; k *= 2)
	{
		// Inner loop: distance between compared elements
		for (size_t j = k / 2; j > 0; j /= 2)
		{
			// Submit a kernel for current (k, j) pass
			q.submit([&](sycl::handler &h)
					 {
                // Get read-write access to buffer data
                auto d = buf.get_access<sycl::access::mode::read_write>(h);

                // Launch parallel kernel with nd_range (global size, local tile size)
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(tile_size)),
                    [=](sycl::nd_item<1> item) {
                        // Get global thread index
                        size_t i = item.get_global_id(0);

                        // Determine index of paired element to compare/swap
                        size_t ixj = i ^ j;

                        // Perform compare-and-swap only if i < ixj and both are in bounds
                        if (ixj > i && ixj < N && i < N) {
                            // Determine sorting direction for this stage
                            bool ascending = ((i & k) == 0);

                            // Load current values
                            int val_i = d[i];
                            int val_ixj = d[ixj];

                            // Swap values if they're out of order according to sorting direction
                            if ((val_i > val_ixj) == ascending) {
                                d[i] = val_ixj;
                                d[ixj] = val_i;
                            }
                        }
                    }
                ); });
		}
	}

	// Wait for all queued kernels to complete before returning
	q.wait();
}

// Compute median from sorted data
long long median(std::vector<long long> &v)
{
	std::sort(v.begin(), v.end());
	size_t n = v.size();
	return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2 : v[n / 2];
}

void print_stats(const std::string &label, const std::vector<long long> &times)
{
	if (times.empty())
		return;

	double sum = std::accumulate(times.begin(), times.end(), 0.0);
	double mean = sum / times.size();

	double sq_sum = 0;
	for (auto t : times)
		sq_sum += (t - mean) * (t - mean);
	double stddev = std::sqrt(sq_sum / times.size());

	long long min_time = *std::min_element(times.begin(), times.end());
	long long max_time = *std::max_element(times.begin(), times.end());

	auto sorted_times = times;
	long long med = median(sorted_times);

	std::cout << "\n--- " << label << " ---\n";
	std::cout << "Avg     : " << mean << " us\n";
	std::cout << "Stddev  : " << stddev << " us\n";
	std::cout << "Min     : " << min_time << " us\n";
	std::cout << "Max     : " << max_time << " us\n";
	std::cout << "Median  : " << med << " us\n";
}

int main()
{
	constexpr int ITER = 10;
	constexpr size_t N = 1 << 20; // 8192 elements

	std::vector<int> orig(N);
	for (size_t i = 0; i < N; ++i)
		orig[i] = rand() % 1000000;

	auto expect = orig;
	std::sort(expect.begin(), expect.end());

	std::vector<long long> times_v1, times_v2, times_v3, times_v4;

	auto data = orig;
	bitonic_sort_v1(data);

	for (int ver = 1; ver <= 4; ++ver)
	{
		for (int it = 0; it < ITER; ++it)
		{
			auto data = orig;
			auto t0 = std::chrono::high_resolution_clock::now();

			if (ver == 1)
				bitonic_sort_v1(data);
			if (ver == 2)
				bitonic_sort_v3(data);
			if (ver == 3)
				bitonic_sort_v3(data);
			if (ver == 4)
				bitonic_sort_v4(data);

			auto t1 = std::chrono::high_resolution_clock::now();

			if (data != expect)
			{
				std::cerr << "Mismatch in v" << ver << "\n";
				return 1;
			}

			long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
			if (ver == 1)
				times_v1.push_back(us);
			if (ver == 2)
				times_v2.push_back(us);
			if (ver == 3)
				times_v3.push_back(us);
			if (ver == 4)
				times_v4.push_back(us);
		}
	}

	print_stats("v1", times_v1);
	print_stats("v2", times_v2);
	print_stats("v3", times_v3);
	print_stats("v4", times_v4);

	auto dump_array = [](const std::string &label, const std::vector<long long> &v)
	{
		std::cout << "\n"
				  << label << "_times = [";
		for (size_t i = 0; i < v.size(); ++i)
			std::cout << v[i] << (i + 1 < v.size() ? ", " : "");
		std::cout << "]\n";
	};

	dump_array("v1", times_v1);
	dump_array("v2", times_v2);
	dump_array("v3", times_v3);
	dump_array("v4", times_v4);

	return 0;
}