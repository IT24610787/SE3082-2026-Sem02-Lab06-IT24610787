#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
 
/* Serial Fibonacci from the README (unchanged) */
int fib(int n) {
  int i, j;
  if (n < 2)
    return n;
  else {
    i = fib(n - 1);
    j = fib(n - 2);
    return i + j;
  }
}
 
int main(int argc, char *argv[]) {
  int n = (argc > 1) ? atoi(argv[1]) : 40;   /* usage: ./fib_serial [n] */
 
  double tstart = omp_get_wtime();
  int result = fib(n);
  double tstop = omp_get_wtime();
  double tcalc = tstop - tstart;             /* seconds */
 
  printf("fib(%d) = %d\n", n, result);
  printf("Serial time: %f s\n", tcalc);
  return 0;
}