// heat3d.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <iostream>
#include <string>
#include "INIReader.h"



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
    thrust::device_vector<int> Uin(size_x), Uout(size_x);

    /* initialisation */
    thrust::fill(Uin.begin(), Uin.end(), 1);

    return 0;
}