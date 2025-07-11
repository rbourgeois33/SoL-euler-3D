struct SineInitFunctor {
  TYPE dx;

  SineInitFunctor(TYPE dx_) : dx(dx_) {}

  __host__ __device__ TYPE operator()(int i) const {
    TYPE x = -dx * myhalf + i * dx;
    // printf("i=%d\n",i);
    return mysin(2 * mypi * x);
  }
};

void initialize(Config &config, thrust::device_vector<TYPE> &Uin, TYPE max_bw) {
  // === Unpack  ===
  int start_x = config.start_x;
  int end_x = config.end_x;
  int Nghost = config.Nghost;
  TYPE dx = config.dx;
  int Nx = config.Nx;

  auto benchmark_sine = benchmark(
      "fill with sine",
      [&]() {
        thrust::transform(
            thrust::counting_iterator<int>(
                start_x), 
            thrust::counting_iterator<int>(end_x), 
            Uin.begin() + Nghost, 
            SineInitFunctor(dx)  
        );
      },
      0, Nx * sizeof(TYPE), max_bw);
}