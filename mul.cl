#define M(name, type) __kernel void mul_##name( \
    const int n,                                \
    const int m,                                \
    const int k,                                \
    __global const type * A,                    \
    __global const type * B,                    \
    __global type * C                           \
) {                                             \
    const int r = get_global_id(0);             \
    const int c = get_global_id(1);             \
    type result = 0;                            \
    for(int i = 0; i < m; i++) {                \
        result += A[r * m + i] * B[i * k + c];  \
    }                                           \
    C[r * k + c] = result;                      \
}

M(f32, float)
M(f64, double)
M(i8, char)
M(i16, short)
M(i32, int)
M(i64, long long)
M(u8, unsigned char)
M(u16, unsigned short)
M(u32, unsigned int)
M(u64, unsigned long long)

#define A(name, type) __kernel void add_##name( \
    const int n,                                \
    const int m,                                \
    const int k,                                \
    __global const type * A,                    \
    __global const type * B,                    \
    __global type * C                           \
) {                                             \
    const int r = get_global_id(0);             \
    const int c = get_global_id(1);             \
    C[r * n + c] = A[r * n + c] + B[r * n + c]; \
}

A(f32, float)
A(f64, double)
A(i8, char)
A(i16, short)
A(i32, int)
A(i64, long long)
A(u8, unsigned char)
A(u16, unsigned short)
A(u32, unsigned int)
A(u64, unsigned long long)