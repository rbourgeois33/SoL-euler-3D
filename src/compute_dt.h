struct ComputeDtFunctor {
  TYPE dx2, alpha, CFL;

  ComputeDtFunctor(TYPE dx2_, TYPE alpha_, TYPE CFL_)
      : dx2(dx2_), alpha(alpha_), CFL(CFL_) {}

  __host__ __device__ TYPE operator()(TYPE ui) const {
    return (ui * CFL * myhalf * dx2 / alpha) /
           ui; // Operation factice pour + tard
  }
};

TYPE compute_dt(Config &config, thrust::device_vector<TYPE> &Uin, TYPE max_bw) {
  // === Unpack  ===
  int Nghost = config.Nghost;
  int Nx = config.Nx;
  TYPE dx2 = config.dx2;
  TYPE alpha = config.alpha;
  TYPE CFL = config.CFL;

  TYPE dt;

  // === Compute dt using transform_reduce ===
  auto benchmark_compute_dt = benchmark(
      "compute_dt",
      [&]() {
        dt = thrust::transform_reduce(Uin.begin() + Nghost, Uin.end() - Nghost,
                                      ComputeDtFunctor(dx2, alpha, CFL), myhuge,
                                      thrust::minimum<TYPE>());
      },
      Nx * sizeof(TYPE), // input size
      1 * sizeof(TYPE),  // output size
      max_bw);

  std::cout << "dt = " << dt << " expected = " << CFL * myhalf * dx2 / alpha
            << std::endl;

  return dt;
}