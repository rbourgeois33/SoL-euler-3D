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

TYPE sum = 0;
auto benchmark_sum = benchmark(
    "sum",
    [&]() {
      sum = thrust::reduce(Uin.begin() + Nghost, Uin.end() - Nghost, myzero,
                           thrust::plus<TYPE>());
    },
    Nx * sizeof(TYPE), 1 * sizeof(TYPE), max_bw);

std::cout << "sum = " << dx * sum << " expected = " << 0.0 << std::endl;