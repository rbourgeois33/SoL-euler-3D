struct TimeStepFunctor {
  TYPE dt_, dx_, alpha_;

  TimeStepFunctor(TYPE dt, TYPE dx, TYPE alpha)
      : dt_(dt), dx_(dx), alpha_(alpha) {}

  __host__ __device__ float
  operator()(thrust::tuple<TYPE, TYPE, TYPE> t) const {
    TYPE Um = thrust::get<0>(t);
    TYPE Uc = thrust::get<1>(t);
    TYPE Up = thrust::get<2>(t);

    // Explicit 1D heat equation stencil
    return Uc + alpha_ * dt_ / (dx_ * dx_) * (Up - 2 * Uc + Um);
  }
};

void time_step(Config &config, thrust::device_vector<TYPE> &Uin,
               thrust::device_vector<TYPE> &Uout, TYPE dt, TYPE max_bw) {

  // === Unpack ===
  int Nghost = config.Nghost;
  int size_x = config.size_x;
  int Nx = config.Nx;
  TYPE dx = config.dx;
  TYPE alpha = config.alpha;

  // Define input zip range: [U[i-1], U[i], U[i+1]]
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(Uin.begin(), Uin.begin() + 1, Uin.begin() + 2));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(Uin.end() - 2, Uin.end() - 1, Uin.end()));

  // Apply the time stepping functor to each stencil
  auto benchmark_timestep = benchmark(
      "time_step",
      [&]() {
        thrust::transform(begin, end,
                          Uout.begin() + Nghost, // skip ghost cells
                          TimeStepFunctor(dt, dx, alpha));
      },
      size_x * sizeof(TYPE), // bytes read
      Nx * sizeof(TYPE),     // bytes written
      max_bw);
}