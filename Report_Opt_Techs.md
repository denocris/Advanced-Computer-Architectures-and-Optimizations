# Advanced Optimization Techniques: Matrix Multiplications

##Introduction and Task

**NOTE:** The source code can be found in the folder `/Ex_matmul_AdvOpt`

The task of this exercise is to perform a matrix multiplication in *C* using different techniques. At the end we will see that a library such that *Lapack cblas*, which in general is extremely optimized, does not have really good performances solving small size problems, due to its overhead which overcomes calculations.

In particular in what follows we collect timing results which corresponds to $10$ iterations of $100000$ matrix multiplication, $c=a \cdot b$ , of $4 \times 4$ matrices.


We will measure performances of the following different approach:

- Naive implementation
- Lapack *cblas_dgemm()*
- Array notation (AN)
- Intrinsic approach with AVX

In the following paragraphs we describe the different approaches and after that we will discuss the results. Before compiling this module are required:

- intel/14.0
- mkl/11.1

#### Naive implementation

The following is the simplest implementation of a matrix multiplication in *C*, using $3$ for loops.
```c
void matrixmul_naive(double* c,double* a,double* b){
    for(int i=0; i<4; i++)
      for(int j=0; j<4; j++)
        for(int k=0; k<4; k++){
          c[i*4 + j] += a[i*4 + k] * b[k*4 + j];
      }
}
```
#### Lapack implementation

The following function was already implemented by Chris Dahnken.
```c
void matrixmul_mnk(double* c,double* a,double* b){
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	      mnk, mnk, mnk, 1, a, mnk, b, mnk, 1, c, mnk);
}
```
$mkn = 4$ is the matrix size. To compile the code it is important to insert the flag `-mkl=sequential` as follows
```bash
icc -mkl=sequential exercise-matmul.cpp
```
This flag is related to the *Lapack* library and `sequential` is required to have the serial implementation. Otherwise the default is `parallel`. Note the presence of the *Intel* compiler `icc`.

#### Array Notation (AN) Implementation
The *Intel* complier allows to use an array notation very similar to *Python* or *Matlab* (https://www.cilkplus.org/tutorial-array-notation). The array notation is introduced to help the compiler utilize SIMD (Single Instruction Multiple Data) instruction on *Intel* architecture CPUs by adding more information about the datastructure and to simplify implementation.

To enable auto-vectorization on *Intel* architecture CPUs the flag `-xAVX` is needed. To enable *Array Notation* we also need the flag `-intel-extensions`.
```bash
icc -mkl=sequential -xAVX -intel-extensions exercise-matmul.cpp
```
The code reads

```c
void matrixmul_mnk_AN(double* c,double* a,double* b){
    for(int i=0; i<4; i++)
      for(int k=0; k<4; k++){
        c[i*4:4]+=a[i*4 + k]*b[k*4:4];
      }
}
```
It is also possible to add the `inline` compiler instruction in front of the void function.
```c
inline void matrixmul_mnk_AN(){ ... }
```
This keyword tells the compiler to substitute the code within the function definition for every instance of a function call (it is exactly as wrinting the instructions without putting them into the function body). This is useful for small functions and it avoids conditional jumps in the assembly code gaining thus performance.

#### AVX Intrinsic Implementation

An intrinsic function is a function available for use in a given programming language whose implementation is handled specially by the compiler and directly gives insructions to registers and ALUs. It is useful to control the work flow inside the CPU and perform parallelized computations inside a single core. The complete guide can be found here (https://software.intel.com/sites/landingpage/IntrinsicsGuide/).

Intrinsics notation can really enhance the performances. However it requires some effort to program it. It is suggested to been used in small and specific parts of the code.

To implement at its best this approach is required to use the aligned version of malloc
```c
double* a= (double*) _mm_malloc(sizeof(double)*size,64);
```

The matrix multiplication code is the following
```c
void matrixmul_intrinsic(double* c, double* a, double* b){
  // Let's create 256bits arrays in cache
  __m256d a_line, b_line, c_line;
      for(int i=0; i<mnk*mnk; i+=4){
        // load rows of c in c_line
        c_line = _mm256_load_pd(&c[i*4]);
        for (int j = 0; j < 4; j++) {
            a_line = _mm256_load_pd(&a[j*4]);
            b_line = _mm256_set1_pd(b[i+j]); // broadcast
            c_line = _mm256_add_pd(_mm256_mul_pd(a_line,b_line), c_line); // multiply and sum
        }
        _mm256_store_pd(&c[i], c_line); // store c_line in c
      }
}
```

### Performance results

Here below a table with the results obtained, after averaging over 5 runs on Ulysses.

-------| Naive | Lapack | AN | Intrinsic
-------|--------|--------|----| -----
time (s) | 0.032 | 0.110 | 0.037 | 0.025
GFLOPs | 3.914 | 1.160 | 3.388 | 5.064

We can easily see that the slowest approach is the *Lapack* implementation. This is because it is optimized for large size problems and in our case, when the size is small, overhead slows down the performances. The same rule is true for the AN implementation, which result slower than the Naive one (on average. In some cases it was slightly higher). The Intrinsic approach instead results by far the fastest.  
