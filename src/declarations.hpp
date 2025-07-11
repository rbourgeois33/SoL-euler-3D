using TYPE = float;

constexpr TYPE mypi = 3.14159265358979323846;
constexpr TYPE myhalf = 0.5;
constexpr TYPE myzero = 0.0;
constexpr TYPE myone = 1.0;
constexpr TYPE myhuge = 1e100;

__host__ __device__ inline TYPE mysin(TYPE x) {
    if constexpr (std::is_same<TYPE, float>::value)
        return sinf(x); 
    else if constexpr (std::is_same<TYPE, double>::value)
        return sin(x);   
}

