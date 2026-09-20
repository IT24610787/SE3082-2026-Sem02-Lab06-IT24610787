#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
 
#define N 1000000
 
/* Strip size in ELEMENTS (floats). It must be a multiple of the SIMD width
    so that each strip is a whole number of vectors:
        SSE      128-bit ->  4 floats
        AVX2     256-bit ->  8 floats
        AVX-512  512-bit -> 16 floats
    1024 is a multiple of all three (and 1024 floats = 4 KB per array, so the
    three strips of A, B, C together fit easily in L1 cache).
    Override at compile time with -DSTRIP=... to try other sizes. */
#ifndef STRIP
#define STRIP 1024
#endif
 
/* Baseline: plain serial loop. Auto-vectorization is switched off for this
    function so it is a true scalar reference (newer GCC versions may otherwise
    vectorize it by themselves at -O2). */
__attribute__((noinline, optimize("no-tree-vectorize")))
static void multiply_serial(const float *restrict a, const float *restrict b,
                            float *restrict c) {
    for (long i = 0; i < N; i++)
        c[i] = a[i] * b[i];
}
 
/* Strip-mined + parallel + SIMD version */
__attribute__((noinline))
static void multiply_strip(const float *restrict a, const float *restrict b,
                           float *restrict c) {
    long nfull = N / STRIP;                 /* number of complete strips */

    /* Outer loop over strips: the strips are shared among the threads. */
    #pragma omp parallel for schedule(static)
    for (long s = 0; s < nfull; s++) {
    long base = s * STRIP;

    /* Inner loop over ONE strip: executed by one thread, using SIMD. */
    #pragma omp simd
    for (long i = 0; i < STRIP; i++)
        c[base + i] = a[base + i] * b[base + i];
    }

    /* Remainder strip: the N % STRIP leftover elements (none if N divides evenly). */
    #pragma omp simd
    for (long i = nfull * STRIP; i < N; i++)
    c[i] = a[i] * b[i];
}
 
int main(int argc, char *argv[]) {
    int reps = (argc > 1) ? atoi(argv[1]) : 100;   /* usage: ./stripmine [reps] */

    size_t bytes = ((N * sizeof(float) + 63) / 64) * 64;   /* multiple of 64 for aligned_alloc */
    float *A = aligned_alloc(64, bytes);
    float *B = aligned_alloc(64, bytes);
    float *C = aligned_alloc(64, bytes);     /* result of the strip-mined version */
    float *R = aligned_alloc(64, bytes);     /* result of the serial version (reference) */
    if (!A || !B || !C || !R) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.5f;
        B[i] = (float)(i % 13) + 1.0f;
        C[i] = 0.0f;
        R[i] = 0.0f;
    }

    /* Warm-up pass: starts the thread team and brings the data into memory/cache */
    multiply_serial(A, B, R);
    multiply_strip(A, B, C);

    /* One pass over 1,000,000 elements takes only milliseconds, so time many
        passes and report the average per pass. */
    double tstart, tstop, tcalc;

    tstart = omp_get_wtime();
    for (int r = 0; r < reps; r++)
        multiply_serial(A, B, R);
    tstop = omp_get_wtime();
    tcalc = tstop - tstart;                          /* seconds for all reps */
    double t_serial = tcalc / reps;

    tstart = omp_get_wtime();
    for (int r = 0; r < reps; r++)
        multiply_strip(A, B, C);
    tstop = omp_get_wtime();
    tcalc = tstop - tstart;
    double t_strip = tcalc / reps;

    /* Verify against the serial result. Each element is a single float
        multiplication, so the results should be bit-identical. */
    long mismatches = 0;
    for (long i = 0; i < N; i++)
        if (C[i] != R[i])
        mismatches++;

    printf("N = %d, strip = %d elements (%zu bytes), full strips = %d, remainder = %d elements\n",
            N, STRIP, STRIP * sizeof(float), N / STRIP, N % STRIP);
    printf("Threads = %d, reps = %d\n", omp_get_max_threads(), reps);
    printf("Serial (scalar):     %.4f ms per pass\n", t_serial * 1000.0);
    printf("Strip-mined OpenMP:  %.4f ms per pass\n", t_strip * 1000.0);
    printf("Speedup:             %.2f\n", t_serial / t_strip);
    printf("Verification:        %s (%ld mismatches)\n",
            mismatches == 0 ? "OK" : "FAILED", mismatches);

    free(A); free(B); free(C); free(R);
    return mismatches != 0;
}