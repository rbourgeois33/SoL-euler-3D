// heat3d.cu
#include "INIReader.h"
#include <iostream>
#include <string>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "declarations.hpp"
#include "benchmark.hpp"

// -----------------------------------------------------------------------------
// Functors
// -----------------------------------------------------------------------------
struct SineInitFunctor {
  TYPE dx;

  __host__ __device__ explicit SineInitFunctor(TYPE dx_) : dx(dx_) {}

  __host__ __device__ TYPE operator()(int i) const {
    TYPE x = -dx * myhalf + i * dx;
    // printf("i=%d\n",i);
    return mysin(2 * mypi * x);
  }
};

struct ComputeDtFunctor {
  TYPE dx2, alpha, CFL;

  __host__ __device__ ComputeDtFunctor(TYPE dx2_, TYPE alpha_, TYPE CFL_)
      : dx2(dx2_), alpha(alpha_), CFL(CFL_) {}

  __host__ __device__ TYPE operator()(TYPE ui) const {
    return (ui * CFL * myhalf * dx2 / alpha) / ui;
  }
};

struct time_step {
  TYPE dt_, dx_, alpha_;

  time_step(TYPE dt, TYPE dx, TYPE alpha) : dt_(dt), dx_(dx), alpha_(alpha) {}

  __host__ __device__ float
  operator()(thrust::tuple<TYPE, TYPE, TYPE> t) const {
    TYPE Um = thrust::get<0>(t);
    TYPE Uc = thrust::get<1>(t);
    TYPE Up = thrust::get<2>(t);

    // Explicit 1D heat equation stencil
    return Uc + alpha_ * dt_ / (dx_ * dx_) * (Up - 2 * Uc + Um);
  }
};

int main(int argc, char *argv[]) {

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
  TYPE CFL = reader.GetReal("simulation", "CFL", -1.0);
  int Nghost = reader.GetInteger("simulation", "Nghost", -1);
  TYPE alpha = reader.GetInteger("simulation", "alpha", 1);

  if (Nx < 0 || CFL < 0.0) {
    std::cerr << "Invalid or missing Nx or CFL in config file." << std::endl;
    return 1;
  }

  TYPE max_bw = get_device_bandwidth_GBs();
  std::cout << "Parsed values:\n";
  std::cout << "Nx = " << Nx << "\n";
  std::cout << "CFL = " << CFL << "\n";
  std::cout << "Nghost = " << Nghost << "\n";

  const int size_x = Nx + 2 * Nghost;
  const int start_x = Nghost;
  const int end_x = size_x - Nghost;
  TYPE dx = 1.0 / (Nx);
  TYPE dx2 = dx * dx;

  /* device allocation */
  thrust::device_vector<TYPE> Uin(size_x), Uout(size_x);

  /* initialisation */
  auto benchmark_fill = benchmark(
      "fill with 1", [&]() { thrust::fill(Uin.begin(), Uin.end(), myone); }, 0,
      size_x * sizeof(TYPE), max_bw);

  TYPE sum_1 = 0;
  auto benchmark_sum_1 = benchmark(
      "sum of 1's",
      [&]() {
        sum_1 = thrust::reduce(Uin.begin() + Nghost, Uin.end() - Nghost, myzero,
                               thrust::plus<TYPE>());
      },
      Nx * sizeof(TYPE), 1 * sizeof(TYPE), max_bw);
  std::cout << "sum_1 = " << sum_1 << " expected = " << (TYPE)Nx << std::endl;

  /* initialisation sine */
  auto benchmark_sine = benchmark(
      "fill with sine",
      [&]() {
        thrust::transform(
            thrust::counting_iterator<int>(start_x), // input First, integers
            thrust::counting_iterator<int>(end_x),   // input Last, integers
            Uin.begin() + Nghost,                    // Result
            SineInitFunctor(dx));
      },
      0, Nx * sizeof(TYPE), max_bw);

  TYPE dtmin = 0;
  auto benchmark_compute_dt = benchmark(
      "compute_dt",
      [&]() {
        dtmin = thrust::transform_reduce(
            Uin.begin() + Nghost, Uin.end() - Nghost,
            ComputeDtFunctor(dx2, alpha, CFL), myhuge, thrust::minimum<TYPE>());
      },
      Nx * sizeof(TYPE), 1 * sizeof(TYPE), max_bw);

  std::cout << "dtmin= " << dtmin << " expected=" << CFL * myhalf * dx2 / alpha
            << std::endl;

  TYPE sum = 0;
  auto benchmark_sum = benchmark(
      "sum",
      [&]() {
        sum = thrust::reduce(Uin.begin() + Nghost, Uin.end() - Nghost, myzero,
                             thrust::plus<TYPE>());
      },
      Nx * sizeof(TYPE), 1 * sizeof(TYPE), max_bw);

  std::cout << "sum = " << dx * sum << " expected = " << 0.0 << std::endl;

  // Time step: define the range of the inputs um uc up, [0,1,2 ---->size_x-2,
  // size_x-1, size_x)
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(Uin.begin(), Uin.begin() + 1, Uin.begin() + 2));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(Uin.end() - 2, Uin.end() - 1, Uin.end() - 0));

  auto benchmark_timestep = benchmark(
      "time_step",
      [&]() {
        thrust::transform(begin, end,
                          Uout.begin() + Nghost, // output skips first point
                          time_step(dtmin, dx, alpha));
      },
      size_x * sizeof(TYPE), Nx * sizeof(TYPE), max_bw);

      return 0;
}