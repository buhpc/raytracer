#if defined (_OPENMP)
#include <omp.h>
#elif defined (USE_MPI)
#include "mpi.h"
#else
#include <sys/time.h>
#endif

#include <iostream>
#include <cstdio>
#include "walltimer.h"

using namespace std;

namespace {

double wallTime()
  {
#if defined _OPENMP
  return omp_get_wtime();
#elif defined (USE_MPI)
  return MPI_Wtime();
#else
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1e-6*t.tv_usec;
#endif
  }

} // unnamed namespace

void wallTimer::start()
  { t_started=wallTime(); running=true; }
void wallTimer::stop()
  { if (running) t_acc+=wallTime()-t_started; running=false; }
double wallTimer::acc() const
  { return running ? t_acc+wallTime()-t_started : t_acc; }

void wallTimerSet::start(const string &name)
  { timers[name].start(); }
void wallTimerSet::stop(const string &name)
  { timers[name].stop(); }
void wallTimerSet::reset(const string &name)
  { timers[name].reset(); }
double wallTimerSet::acc(const string &name)
  { return timers[name].acc(); }

void wallTimerSet::report() const
  {
  cout << "\nWall clock timer report:" << endl;
  for (map<string,wallTimer>::const_iterator it=timers.begin(); it!=timers.end(); ++it)
    printf("  %-15s: %10.5fs\n", it->first.c_str(), it->second.acc());
  cout << "End wall clock timer report\n" << endl;
  }

wallTimerSet wallTimers;
