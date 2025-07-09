// heat3d.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <iostream>
#include <string>
#include "INIReader.h"

using _TYPE_ = double;

struct BenchmarkResult {
  float milliseconds;
  double flops;
  double bandwidth_bytes;
};

inline BenchmarkResult benchmark(const std::string& label,
                                 std::function<void()> func,
                                 size_t flops,
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

  float ms = std::chrono::duration<float, std::milli>(end - start).count();

  double gflops = nrepeat*(flops / 1e9) / (ms / 1000.0); //GFLOPs;
  double bandwidth = nrepeat*((loads + stores) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0); // GB/s

  std::cout <<"\n==" <<label << "==  "<<"  Time: " << ms << " ms\n";
  if (flops > 0)
    std::cout << "  Throughput:" << gflops << " GFLOP/s\n";
  if (loads + stores > 0)
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
  std::cout << std::endl;

  return {ms, gflops, bandwidth};
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " config.ini" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    INIReader reader(filename);

    if (reader.ParseError() != 0) {
        std::cerr << "Can't load or parse " << filename << std::endl;
        return 1;
    }

    int Nx = reader.GetInteger("simulation", "Nx", -1);
    double CFL = reader.GetReal("simulation", "CFL", -1.0);
    int Nghost = reader.GetInteger("simulation", "Nghost", -1);


    if (Nx < 0 || CFL < 0.0) {
        std::cerr << "Invalid or missing Nx or CFL in config file." << std::endl;
        return 1;
    }

    std::cout << "Parsed values:\n";
    std::cout << "Nx = " << Nx << "\n";
    std::cout << "CFL = " << CFL << "\n";   
    std::cout << "Nghost = " << Nghost << "\n";   

    const int size_x = Nx + 2*Nghost;


    /* device allocation */
    thrust::device_vector<_TYPE_> Uin(size_x), Uout(size_x);

    /* initialisation */
    auto benchmark_fill = benchmark("fill", [&]() {
      thrust::fill(Uin.begin(), Uin.end(), 1);}, 
      0, //#of operations
      0, //# of reads
      size_x*sizeof(_TYPE_)); //# of writes

    return 0;
}