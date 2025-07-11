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

#include "declarations.h"
#include "config.h"
#include "benchmark.h"
#include "compute_dt.h"
#include "initialize.h"
#include "time_step.h"
#include "boundary_conditions.h"

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " config.ini" << std::endl;
    return 1;
  }

  std::string filename = argv[1];

  Config config(filename);
  config.print();

  TYPE max_bw = get_device_bandwidth_GBs();

  /* device allocation */
  thrust::device_vector<TYPE> Uin(config.size_x), Uout(config.size_x);

  initialize(config, Uin, max_bw);
  TYPE dt = compute_dt(config, Uin, max_bw);
  config.set_dt(dt);

  TYPE tmax = config.tmax;
  int Nstepmax = config.Nstepmax;

  TYPE t=0;
  int Nstep = 0;
  bool should_stop = false;

  while((t<tmax)&&(Nstep<Nstepmax)&&(not(should_stop))){

    if (t+dt > tmax){
      dt = tmax - t;
      std::cout<<"\nClamping dt for the last time step !\n";
      should_stop = true;
    }

    if (Nstep+1 == Nstepmax){
      std::cout<<"\nMax Nstep reached, last time step !\n";
      should_stop = true;
    }

    boundary_condition(config, Uin);
    time_step(config, Uin, Uout, dt, max_bw);
    thrust::swap(Uin, Uout);
    t += dt;
    Nstep += 1;
  }
  
  

  return 0;
}