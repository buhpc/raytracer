/*
Copyright things go here.
*/

#ifndef SPLOTCH_CUDA_H
#define SPLOTCH_CUDA_H

#ifdef VS
#define THREADFUNC DWORD WINAPI
#else
#define THREADFUNC static void*
#endif

#include <cstring>
#include "cxxsupport/paramfile.h"
#include "kernel/colour.h"
#include "splotch/splotchutils.h"
class CuPolicy;

//data structs for using on device
//'d_' means device
typedef particle_sim cu_particle_sim;

#define MAX_P_TYPE 8//('XXXX','TEMP','U','RHO','MACH','DTEG','DISS','VEL')
                                        //in mid of developing only
#define MAX_EXP -20.0

struct cu_param_range //parameters for range calculation
  {
  int ptypes; //meaning how many types
  bool col_vector[MAX_P_TYPE], log_int[MAX_P_TYPE],
    log_col[MAX_P_TYPE], asinh_col[MAX_P_TYPE];
  float mincol[MAX_P_TYPE], maxcol[MAX_P_TYPE],
    minint[MAX_P_TYPE], maxint[MAX_P_TYPE];
  };

struct cu_param_transform
  {
  float p[12];
  bool  projection;
  int   xres, yres;
  float fovfct, dist, xfac;
  float minrad_pix;
  };

struct cu_color
  {
  float r,g,b;
  };

struct cu_color_map_entry
  {
  float val;
  cu_color color;
  };

struct cu_colormap_info
  {
  cu_color_map_entry *map;
  int mapSize;
  int *ptype_points;
  int ptypes;
  };

struct cu_particle_splotch
  {
  float x,y,r,ro;
  cu_color e;
  bool isValid;
  unsigned short minx, miny, maxx, maxy;
  unsigned long posInFragBuf;
  };

struct cu_param_colorize
  {
  int xres, yres, ptypes;
  float zmaxval, zminval;
  bool col_vector[MAX_P_TYPE];
  float brightness[MAX_P_TYPE];
  float rfac;
  };

struct cu_exptable_info
  {
  float expfac, taylorlimit;
  float *tab1, *tab2;
  enum {
    nbits=10,
    dim1=1<<nbits,
    mask1=dim1-1,
    dim2=1<<nbits<<nbits,
    mask2=dim2-1,
    mask3=~mask2
    };
  };

struct cu_fragment_AeqE
  {
  float aR, aG, aB;
  };

struct cu_fragment_AneqE
  {
  float aR, aG, aB;
  float qR, qG, qB;
  };


struct cu_gpu_vars //variables used by each gpu
  {
  CuPolicy            *policy;
  cu_particle_sim     *d_pd;             //device_particle_data
  cu_colormap_info    d_colormap_info;   //device pointers
  cu_exptable_info    d_exp_info;        //device pointers
  cu_particle_splotch *d_ps_render;
  void                *d_fbuf;
  cu_color            *d_pic;
  };

//functions

void cu_init(int devID, int nP, cu_gpu_vars* pgv);
void cu_copy_particles_to_device(cu_particle_sim* h_pd, unsigned int n, cu_gpu_vars* pgv);
size_t cu_allocate_fbuf(int nP, cu_gpu_vars* pgv);
void cu_range(paramfile &params, cu_particle_sim* p, unsigned int n, cu_gpu_vars* pgv);
void cu_transform (paramfile &fparams, unsigned int n,
  vec3 &campos, vec3 &lookat, vec3 &sky, cu_particle_sim* h_pd, cu_gpu_vars* pgv);
void cu_init_colormap(cu_colormap_info info, cu_gpu_vars* pgv);
void cu_colorize(paramfile &params, cu_particle_splotch *h_ps, int n, cu_gpu_vars* pgv);
void cu_init_exptab(double maxexp, cu_gpu_vars* pgv);
void cu_realloc_particles_to_render(int n, cu_gpu_vars* pgv);
void cu_copy_particles_to_render(cu_particle_splotch *p,int n, cu_gpu_vars* pgv);
void cu_render1(int nP, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv);
void cu_get_fbuf (void *h_fbuf, bool a_eq_e, unsigned long n, cu_gpu_vars* pgv);
void cu_end (cu_gpu_vars* pgv);
int cu_get_chunk_particle_count(paramfile &params, CuPolicy* policy);
void getCuTransformParams(cu_param_transform &para_trans,
paramfile &params, vec3 &campos, vec3 &lookat, vec3 &sky);


#endif
