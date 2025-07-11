struct BenchmarkResult {
  TYPE milliseconds;
  TYPE flops;
  TYPE bandwidth_bytes;
};

inline BenchmarkResult benchmark(const std::string& label,
                                 std::function<void()> func,
//                                 size_t flops,
                                 size_t loads,
                                 size_t stores,
                                 size_t nrepeat=1)
{
  // Sync before
  cudaDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();
  
  for (size_t n=0; n<nrepeat ; n++){
  func();
  cudaDeviceSynchronize();
  }

  auto end = std::chrono::high_resolution_clock::now();

  TYPE ms = std::chrono::duration<TYPE, std::milli>(end - start).count();

 // TYPE gflops = nrepeat*(flops / 1e9) / (ms / 1000.0); //GFLOPs;
  TYPE bandwidth = nrepeat*((loads + stores) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0); // GB/s

  std::cout <<"\n==" <<label << "==  "<<"  Time: " << ms << " ms\n";
 // if (flops > 0)
 //   std::cout << "  Throughput:" << gflops << " GFLOP/s\n";
  if (loads + stores > 0)
    std::cout << "  Bandwidth: " << bandwidth << " GB/s";
  std::cout << std::endl;

  return {ms, 0, bandwidth};
}