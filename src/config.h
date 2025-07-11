struct Config {
  // Simulation parameters
  int Nx;
  TYPE CFL;
  int Nghost;
  TYPE alpha;
  TYPE tmax;
  int Nstepmax;

  // Derived grid parameters
  int size_x;
  int start_x;
  int end_x;
  TYPE dx;
  TYPE dx2;
  TYPE invdx2;

  TYPE dt = -1.0;

  // Constructor to initialize all fields
  Config(std::string &filename) {

    INIReader reader(filename);
    Nx = reader.GetInteger("simulation", "Nx", -1);
    CFL = reader.GetReal("simulation", "CFL", -1.0);
    Nghost = reader.GetInteger("simulation", "Nghost", -1);
    alpha =
        reader.GetReal("simulation", "alpha", 1.0); // corrected from GetInteger
    tmax =
        reader.GetReal("simulation", "tmax", 1.0); // corrected from GetInteger
    Nstepmax = reader.GetInteger("simulation", "Nstepmax", -1);

    size_x = Nx + 2 * Nghost;
    start_x = Nghost;
    end_x = size_x - Nghost;

    dx = TYPE(1.0) / Nx;
    dx2 = dx * dx;
    invdx2 = TYPE(1.0) / dx2;
  }

  void print() const {
    std::cout << "Parsed values:\n";
    std::cout << "Nx = " << Nx << "\n";
    std::cout << "CFL = " << CFL << "\n";
    std::cout << "Nghost = " << Nghost << "\n";
    std::cout << "alpha = " << alpha << "\n";
    std::cout << "tmax = " << tmax << "\n";
    std::cout << "Nstepmax = " << Nstepmax << "\n";
    std::cout << "dt = " << dt << "\n";
    std::cout << "dx = " << dx << ", dx^2 = " << dx2 << ", 1/dx^2 = " << invdx2
              << "\n";
  }

  void set_dt(TYPE dt_){
    dt=dt_;
  }
};