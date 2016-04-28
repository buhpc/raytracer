/*
Try accelerating splotch with CUDA. July 2009.
Copyright things go here.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cutil_inline.h>

#include "splotch/splotchutils.h"
#include "kernel/transform.h"

#include "cuda/splotch_kernel.cu"
#include "cuda/splotch_cuda.h"
#include "cuda/CuPolicy.h"


using namespace std;

template<typename T> T findParamWithoutChange
  (paramfile *param, std::string &key, T &deflt)
  {
  return param->param_present(key) ? param->find<T>(key) : deflt;
  }

#define CLEAR_MEM(p) if(p) {cutilSafeCall(cudaFree(p)); p=0;}


void getCuTransformParams(cu_param_transform &para_trans,
paramfile &params, vec3 &campos, vec3 &lookat, vec3 &sky)
  {
  int xres = params.find<int>("xres",800),
      yres = params.find<int>("yres",xres);
  double fov = params.find<double>("fov",45); //in degrees
  double fovfct = tan(fov*0.5*degr2rad);
  float64 xfac=0.0, dist=0.0;

  sky.Normalize();
  vec3 zaxis = (lookat-campos).Norm();
  vec3 xaxis = crossprod (sky,zaxis).Norm();
  vec3 yaxis = crossprod (zaxis,xaxis);
  TRANSFORM trans;
  trans.Make_General_Transform
        (TRANSMAT(xaxis.x,xaxis.y,xaxis.z,
                  yaxis.x,yaxis.y,yaxis.z,
                  zaxis.x,zaxis.y,zaxis.z,
                  0,0,0));
  trans.Invert();
  TRANSFORM trans2;
  trans2.Make_Translation_Transform(-campos);
  trans2.Add_Transform(trans);
  trans=trans2;
  bool projection = params.find<bool>("projection",true);

  if (!projection)
    {
    float64 dist= (campos-lookat).Length();
    float64 xfac=1./(fovfct*dist);
    cout << " Field of fiew: " << 1./xfac*2. << endl;
    }

  float minrad_pix = params.find<float>("minrad_pix",1.);

  //retrieve the parameters for transformation
  for (int i=0; i<12; i++)
    para_trans.p[i] =trans.Matrix().p[i];
  para_trans.projection=projection;
  para_trans.xres=xres;
  para_trans.yres=yres;
  para_trans.fovfct=fovfct;
  para_trans.dist=dist;
  para_trans.xfac=xfac;
  para_trans.minrad_pix=minrad_pix;
  }


void cu_init(int devID, int nP, cu_gpu_vars* pgv)
  {
  cudaSetDevice (devID); // initialize cuda runtime
  
  //allocate device memory for particle data
  size_t s = pgv->policy->GetSizeDPD(nP);
  //one more space allocated for the dumb
  cutilSafeCall(cudaMalloc((void**) &pgv->d_pd, s +sizeof(cu_particle_sim)));
  
  //now prepare memory for d_particle_splotch.
  //one more for dums
  s = nP* sizeof(cu_particle_splotch);
  cutilSafeCall( cudaMalloc((void**) &pgv->d_ps_render, s+sizeof(cu_particle_splotch)));

  size_t size = pgv->policy->GetFBufSize() <<20;
  cutilSafeCall( cudaMalloc((void**) &pgv->d_fbuf, size)); 
  }


void cu_copy_particles_to_device(cu_particle_sim* h_pd, unsigned int n, cu_gpu_vars* pgv)
  {
  //copy particle data to device
  size_t s = pgv->policy->GetSizeDPD(n);
  cutilSafeCall(cudaMemcpy(pgv->d_pd, h_pd, s, cudaMemcpyHostToDevice) );
  }

void cu_range(paramfile &params ,cu_particle_sim* h_pd,
  unsigned int n, cu_gpu_vars* pgv)
  {

  //prepare parameters for stage 1
  cu_param_range pr;
  pr.ptypes = params.find<int>("ptypes",1);
  //now collect parameters from configuration
  for(int itype=0;itype<pr.ptypes;itype++)
    {
    pr.log_int[itype] = params.find<bool>("intensity_log"+dataToString(itype),true);
    pr.log_col[itype] = params.find<bool>("color_log"+dataToString(itype),true);
    pr.asinh_col[itype] = params.find<bool>("color_asinh"+dataToString(itype),false);
    pr.col_vector[itype] = params.find<bool>("color_is_vector"+dataToString(itype),false);
    pr.mincol[itype]=1e30;
    pr.maxcol[itype]=-1e30;
    pr.minint[itype]=1e30;
    pr.maxint[itype]=-1e30;
    }
  //allocate memory on device and dump parameters to it
  cu_param_range  *d_pr = 0;
  int s = sizeof(cu_param_range);
  cutilSafeCall( cudaMalloc((void**) &d_pr, s) );
  cutilSafeCall( cudaMemcpy(d_pr, &pr, s, cudaMemcpyHostToDevice) );

  //ask for dims from pgv->policy
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(n, &dimGrid, &dimBlock);
  // call device for stage 1
  k_range1<<<dimGrid,dimBlock>>>(d_pr, pgv->d_pd, n);

  // call device for stage 2 ptypes times
  // prepare parameters1 first
  float minval_int, maxval_int, minval_col, maxval_col;
  std::string tmp;
  for(int itype=0;itype<pr.ptypes;itype++)
    {
    tmp = "intensity_min"+dataToString(itype);
    minval_int =findParamWithoutChange<float>(&params,  //in mid of developing only
      tmp, pr.minint[itype]);
    tmp = "intensity_max"+dataToString(itype);
    maxval_int = findParamWithoutChange<float>(&params, tmp, pr.maxint[itype]);
    tmp = "color_min"+dataToString(itype);
    minval_col = findParamWithoutChange<float>(&params, tmp, pr.mincol[itype]);
    tmp = "color_max"+dataToString(itype);
    maxval_col = findParamWithoutChange<float>(&params, tmp, pr.maxcol[itype]);

    k_range2<<<dimGrid, dimBlock>>>(d_pr, pgv->d_pd, n, itype,
      minval_int,maxval_int,minval_col,maxval_col);
    }

  // copy result out to host in mid of development only!!!
  // s = pgv->policy->GetSizeDPD(n);
  // cutilSafeCall(cudaMemcpy(h_pd, pgv->d_pd, s, cudaMemcpyDeviceToHost) );

  //free parameters on device
  CLEAR_MEM((d_pr));
  }


void cu_transform (paramfile &fparams, unsigned int n,
  vec3 &campos, vec3 &lookat, vec3 &sky, cu_particle_sim* h_pd, cu_gpu_vars* pgv)
  {
  //retrieve parameters for transformation first
  cu_param_transform tparams;
  getCuTransformParams(tparams,fparams,campos,lookat,sky);

  //arrange memory for the parameters and copy to device
  cu_param_transform  *d_pt;
  int size = sizeof(cu_param_transform);
  cutilSafeCall( cudaMalloc((void**) &d_pt, size) );
  cutilSafeCall(cudaMemcpy(d_pt, &tparams, size, cudaMemcpyHostToDevice) );

  //Get block dim and grid dim from pgv->policy object
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(n, &dimGrid, &dimBlock);

  //call device transformation
  k_transform<<<dimGrid,dimBlock>>>(pgv->d_pd, n, d_pt);

  //free parameters' device memory
  CLEAR_MEM((d_pt));

  //copy result out to host in mid of development only!!!
  //size =pgv->policy->GetSizeDPD(n);
  //cutilSafeCall(cudaMemcpy(h_pd, pgv->d_pd, size, cudaMemcpyDeviceToHost) );
  }

void cu_init_colormap(cu_colormap_info h_info, cu_gpu_vars* pgv)
  {
  //allocate memories for colormap and ptype_points and dump host data into it
  size_t size =sizeof(cu_color_map_entry) *h_info.mapSize;
  cutilSafeCall( cudaMalloc((void**) &pgv->d_colormap_info.map, size));
  cutilSafeCall(cudaMemcpy(pgv->d_colormap_info.map, h_info.map,
    size, cudaMemcpyHostToDevice) );
  //type
  size =sizeof(int) *h_info.ptypes;
  cutilSafeCall( cudaMalloc((void**) &pgv->d_colormap_info.ptype_points, size));
  cutilSafeCall(cudaMemcpy(pgv->d_colormap_info.ptype_points, h_info.ptype_points,
    size, cudaMemcpyHostToDevice) );

  //set fields of global varible pgv->d_colormap_info
  pgv->d_colormap_info.mapSize =h_info.mapSize;
  pgv->d_colormap_info.ptypes  =h_info.ptypes;
  }

void cu_colorize(paramfile &params, cu_particle_splotch *h_ps,
  int n, cu_gpu_vars* pgv)
  {
  //fetch parameters for device calling first
  cu_param_colorize   pcolorize;
  pcolorize.xres       = params.find<int>("xres",800);
  pcolorize.yres       = params.find<int>("yres",pcolorize.xres);
  pcolorize.zmaxval   = params.find<float>("zmax",1.e23);
  pcolorize.zminval   = params.find<float>("zmin",0.0);
  pcolorize.ptypes    = params.find<int>("ptypes",1);

  for(int itype=0; itype<pcolorize.ptypes; itype++)
    {
    pcolorize.brightness[itype] = params.find<double>("brightness"+dataToString(itype),1.);
    pcolorize.col_vector[itype] = params.find<bool>("color_is_vector"+dataToString(itype),false);
    }
  pcolorize.rfac=1.5;

  //prepare memory for parameters and dump to device
  cu_param_colorize   *d_param_colorize;
  cutilSafeCall( cudaMalloc((void**) &d_param_colorize, sizeof(cu_param_colorize)));
  cutilSafeCall( cudaMemcpy(d_param_colorize, &pcolorize, sizeof(cu_param_colorize),
    cudaMemcpyHostToDevice));

  //fetch grid dim and block dim and call device
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(n, &dimGrid, &dimBlock);
  k_colorize<<<dimGrid,dimBlock>>>(d_param_colorize, pgv->d_pd, n, pgv->d_ps_render,pgv->d_colormap_info);

  //copy the result out
  size_t size = n* sizeof(cu_particle_splotch);
  cutilSafeCall(cudaMemcpy(h_ps, pgv->d_ps_render, size, cudaMemcpyDeviceToHost) );

  //free params memory
  CLEAR_MEM((d_param_colorize));

  //particle_splotch memory on device will be freed in cu_end
  }
 

void cu_init_exptab(double maxexp, cu_gpu_vars* pgv)
  {
  //set common fields of pgv->d_exp_info
  pgv->d_exp_info.expfac =pgv->d_exp_info.dim2 / maxexp;
  //now make up tab1 and tab2 in host
  int dim1 =pgv->d_exp_info.dim1, dim2 =pgv->d_exp_info.dim2;
  float *h_tab1 =new float[dim1];
  float *h_tab2 =new float[dim2];
  for (int m=0; m<dim1; ++m)
    {
    h_tab1[m]=exp(m*dim1/pgv->d_exp_info.expfac);
    h_tab2[m]=exp(m/pgv->d_exp_info.expfac);
    }
  pgv->d_exp_info.taylorlimit = sqrt(2.*abs(maxexp)/dim2);

  //allocate device memory and dump
  size_t size =sizeof(float) *dim1;
  cutilSafeCall( cudaMalloc((void**) &pgv->d_exp_info.tab1, size));
  cutilSafeCall( cudaMemcpy(pgv->d_exp_info.tab1, h_tab1, size,
    cudaMemcpyHostToDevice));
  size =sizeof(float) *dim2;
  cutilSafeCall( cudaMalloc((void**) &pgv->d_exp_info.tab2, size));
  cutilSafeCall( cudaMemcpy(pgv->d_exp_info.tab2, h_tab2, size,
    cudaMemcpyHostToDevice));

  //delete tab1 and tab2 in host
  delete []h_tab1;
  delete []h_tab2;
  }

void cu_copy_particles_to_render(cu_particle_splotch *p,
  int n, cu_gpu_vars* pgv)
  {
  //copy filtered particles into device
  size_t size = n *sizeof(cu_particle_splotch);
  cutilSafeCall(cudaMemcpy(pgv->d_ps_render, p,size,
    cudaMemcpyHostToDevice) );
  }

void cu_render1
  (int nP, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv)
  {
  //endP actually exceed the last one to render
  //get dims from pgv->policy object first
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(nP, &dimGrid, &dimBlock);

  //call device
  k_render1<<<dimGrid, dimBlock>>>(pgv->d_ps_render, nP,
    pgv->d_fbuf, a_eq_e, grayabsorb,pgv->d_exp_info);
  }


void cu_get_fbuf
  (void *h_fbuf, bool a_eq_e, unsigned long n, cu_gpu_vars* pgv)
  {
  size_t size;
  if (a_eq_e)
    size =n* sizeof(cu_fragment_AeqE);
  else
    size =n* sizeof(cu_fragment_AneqE);

  cutilSafeCall( cudaMemcpy(h_fbuf, pgv->d_fbuf,size,
    cudaMemcpyDeviceToHost)) ;
  }

void cu_end(cu_gpu_vars* pgv)
  {
  CLEAR_MEM((pgv->d_pd));
  CLEAR_MEM((pgv->d_ps_render));
  CLEAR_MEM((pgv->d_colormap_info.map));
  CLEAR_MEM((pgv->d_colormap_info.ptype_points));
  CLEAR_MEM((pgv->d_exp_info.tab1));
  CLEAR_MEM((pgv->d_exp_info.tab2));
  CLEAR_MEM((pgv->d_fbuf));

  cudaThreadExit();

  delete pgv->policy;
  }

int cu_get_chunk_particle_count(paramfile &params, CuPolicy* policy)
  {
   int gMemSize = policy->GetGMemSize();
   int fBufSize = policy->GetFBufSize();
   if (gMemSize <= fBufSize) return 0;

   float factor =params.find<float>("particle_mem_factor", 3);
   int spareMem = 10;
   int arrayParticleSize = gMemSize - fBufSize - spareMem;

   return (int) (arrayParticleSize/sizeof(cu_particle_sim)/factor)*(1<<20);
  }
