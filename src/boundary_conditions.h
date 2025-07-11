void boundary_condition(Config &config, thrust::device_vector<TYPE> &Uin){
  // int Nx = config.Nx;
  // int Nghost = config.Nghost;
  // int size_x = config.size_x;

  // U_0 = U_size_x-2
  thrust::copy(Uin.end()-1, Uin.end()-1 + 1, Uin.begin());

  //U_size_x-1 = U_1
  thrust::copy(Uin.begin() + 1, Uin.begin() + 1, Uin.end()-2);
}
