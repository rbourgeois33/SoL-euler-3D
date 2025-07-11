struct BenchmarkResult {
  TYPE milliseconds;
  TYPE flops;
  TYPE bandwidth_bytes;
};

inline BenchmarkResult
benchmark(const std::string &label, std::function<void()> func,
          //                                 size_t flops,
          size_t loads, size_t stores, TYPE max_bw, size_t nrepeat = 1) {
  // Sync before
  cudaDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t n = 0; n < nrepeat; n++) {
    func();
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::high_resolution_clock::now();

  TYPE ms = std::chrono::duration<TYPE, std::milli>(end - start).count();

  // TYPE gflops = nrepeat*(flops / 1e9) / (ms / 1000.0); //GFLOPs;
  TYPE bandwidth = nrepeat * ((loads + stores) / (1024.0 * 1024.0 * 1024.0)) /
                   (ms / 1000.0); // GB/s

  std::cout << "\n==" << label << "==  " << "  Time: " << ms << " ms\n";
  // if (flops > 0)
  //   std::cout << "  Throughput:" << gflops << " GFLOP/s\n";
  if (loads + stores > 0)
    std::cout << "  Bandwidth: " << bandwidth
              << " GB/s = " << 100 * bandwidth / max_bw << "% SoL\n";

  return {ms, 0, bandwidth};
}

#include <cuda_runtime.h>
#include <iostream>

TYPE get_device_bandwidth_GBs() {
  cudaDeviceProp prop;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  // Compute memory bandwidth:
  // memoryClockRate is in kHz, memoryBusWidth is in bits
  TYPE mem_clock_mhz = prop.memoryClockRate * 1e-3; // MHz
  TYPE bus_width_bytes = prop.memoryBusWidth / 8.0; // bits to bytes
  TYPE bandwidth_bytes_per_sec = 2.0 * mem_clock_mhz * 1e6 * bus_width_bytes;

  TYPE bandwidth_GBs = bandwidth_bytes_per_sec / 1e9;

  std::cout << "\nYour device (" << prop.name << ") has a max bandwidth of "
            << bandwidth_GBs << " GB/s" << std::endl;

  return bandwidth_GBs;
}