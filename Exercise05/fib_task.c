#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
 
/* Plain serial version, used below the cutoff */
int fib_serial(int n) {
  int i, j;
  if (n < 2)
    return n;
  i = fib_serial(n - 1);
  j = fib_serial(n - 2);
  return i + j;
}
 
/* Task-parallel version */
int fib_task(int n, int cutoff) {
  int i, j;
 
  if (n < 2)
    return n;
 
  /* Cutoff: small subproblems are cheaper to compute directly than to
     wrap in tasks, so switch back to the serial code. */
  if (n < cutoff)
    return fib_serial(n);
 
  #pragma omp task shared(i)
  i = fib_task(n - 1, cutoff);
 
  #pragma omp task shared(j)
  j = fib_task(n - 2, cutoff);
 
  #pragma omp taskwait          /* wait for both child tasks before adding */
  return i + j;
}
 
int main(int argc, char *argv[]) {
  int n      = (argc > 1) ? atoi(argv[1]) : 40;   /* usage: ./fib_task [n] [cutoff] */
  int cutoff = (argc > 2) ? atoi(argv[2]) : 20;
  int result = 0;
 
  double tstart = omp_get_wtime();
 
  #pragma omp parallel
  {
    #pragma omp single          /* ONE thread creates the root task; the team executes the tasks */
    result = fib_task(n, cutoff);
  }
 
  double tstop = omp_get_wtime();
  double tcalc = tstop - tstart;                  /* seconds */
 
  printf("fib(%d) = %d\n", n, result);
  printf("Threads: %d, cutoff: %d, time: %f s\n", omp_get_max_threads(), cutoff, tcalc);
  return 0;
}