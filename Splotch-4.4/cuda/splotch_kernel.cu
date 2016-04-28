/*
Try accelerating splotch with CUDA. July 2009.
Copyright things go here.
*/

//#include "splotch_kernel.h"
#include "splotch_cuda.h"

//MACROs
#define Pi 3.14159265358979323846264338327950288
#define get_xy_from_sn(sn, xmin, ymin, ymax, x, y)\
        {int x1 =sn/(ymax-ymin); int y1 =sn-x1*(ymax-ymin);\
         x  =x1 +xmin; y  =y1 +ymin;}
#define get_sn_from_xy(x,y,maxy,miny, sn)\
    {sn =x*(maxy-miny) +y;}

#define get_minmax(minv, maxv, val) \
         minv=min(minv,val); \
         maxv=max(maxv,val);

/////////help functions///////////////////////////////////
__device__ float    my_asinh(float val)
  {
  return log(val+sqrt(1.+val*val));
  }

__device__ void my_normalize(float minv, float maxv, float &val)
  {
  if (minv!=maxv) val =  (val-minv)/(maxv-minv);
  }

__device__ void clamp (float minv, float maxv, float &val)
  {
  val = min(maxv, max(minv, val));
  }

//fetch a color from color table on device
__device__ cu_color get_color
  (int ptype, float val, cu_colormap_info info)
  {
  //copy things to local block memory
  __shared__ cu_color_map_entry *map;
  __shared__ int      mapSize;
  __shared__ int *ptype_points;
  __shared__ int ptypes;

  map =info.map;
  mapSize =info.mapSize;
  ptype_points =info.ptype_points;
  ptypes  =info.ptypes;

  cu_color        clr;
  clr.r =clr.g =clr.b =0.0;

  //first find the right entry for this ptype
  if (ptype>=ptypes)
    return clr; //invalid input
  int     start, end;
  start =ptype_points[ptype];
  if ( ptype == ptypes-1)//the last type
    end =mapSize-1;
  else
    end =ptype_points[ptype+1]-1;

  //search the section of this type to find the val
  int i=start;
  while ((val>map[i+1].val) && (i<end))
    ++i;

  const float fract = (val-map[i].val)/(map[i+1].val-map[i].val);
  cu_color clr1=map[i].color, clr2=map[i+1].color;
  clr.r =clr1.r + fract*(clr2.r-clr1.r);
  clr.g =clr1.g + fract*(clr2.g-clr1.g);
  clr.b =clr1.b + fract*(clr2.b-clr1.b);

  return clr;
  }

__device__  float get_exp(float arg, cu_exptable_info d_exp_info)
  {
#if 0
  return exp(arg);
#else
  //fetch things to local
  __shared__  float   expfac;
  __shared__  float   *tab1, *tab2;
  __shared__  int     mask1, mask3, nbits;
  expfac  = d_exp_info.expfac;
  tab1    = d_exp_info.tab1;
  tab2    = d_exp_info.tab2;
  mask1   = d_exp_info.mask1;
  mask3   = d_exp_info.mask3;
  nbits   = d_exp_info.nbits;

  int iarg= (int)(arg*expfac);
  //  for final device code
  if (iarg&mask3)
    return (iarg<0) ? 1. : 0.;
  return tab1[iarg>>nbits]*tab2[iarg&mask1];
#endif
  }

__device__  float get_expm1(float arg, cu_exptable_info d_exp_info)
  {
#if 0
  return exp(arg)-1.0;
#else
  //fetch things to local
  __shared__  float  taylorlimit;
  taylorlimit = d_exp_info.taylorlimit;
  
  if(abs(arg) < taylorlimit) return arg;
  return get_exp(arg, d_exp_info)-1.0;   // exp(x)-1~x
#endif
  }

__global__ void k_post_process(cu_color *pic, int n, cu_exptable_info exp_info)
  {
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >=n)
    m =n;

  //each pic[m] should do the same calc, so sequence does not matter!
  pic[m].r =1.0 -get_exp( pic[m].r, exp_info);
  pic[m].g =1.0 -get_exp( pic[m].g, exp_info);
  pic[m].b =1.0 -get_exp( pic[m].b, exp_info);
  }

__global__ void k_combine
  (int minx, int miny, int maxx, int maxy, int xres, int yres,
  cu_particle_splotch *p, int pStart, int pEnd, cu_fragment_AeqE *fbuf, cu_color *pic)
  {
  int m =blockIdx.x *blockDim.x + threadIdx.x;
  int n =(maxx-minx)*(maxy-miny);
  if (m >=n)
    m =n;

  //get global coordinate point(x,y) of this thread
  int point_x, point_y;
  get_xy_from_sn(m, minx, miny, maxy, point_x, point_y);

  //go through all particles, for each particle p if point(x,y) is in its region
  //p(minx,miny, maxx,maxy) do the following.
  //find the sequencial number sn1 in p(minx,miny, maxx,maxy), the fragment we are looking
  //for in fragment buffer is fragBuf[ sn1+p.posInFBuf ]
  //grab the fragment f(deltaR,deltaG,deltaB)
  //find the sequencial number sn2 of point(x,y) in the output pic.
  //pic[sn2] += f
  int sn1, sn2, local_x, local_y, fpos;
  for (int i=pStart; i<=pEnd; i++)
    {
    if ( point_x >=p[i].minx && point_x<p[i].maxx &&
         point_y >=p[i].miny && point_y<p[i].maxy)
      {
      local_x =point_x -p[i].minx;
      local_y =point_y -p[i].miny;
      get_sn_from_xy(local_x, local_y, p[i].maxy, p[i].miny,sn1);
      fpos =sn1 +p[i].posInFragBuf;

      get_sn_from_xy(point_x, point_y, yres,0, sn2);
      pic[sn2].r +=fbuf[fpos].aR;
      pic[sn2].g +=fbuf[fpos].aG;
      pic[sn2].b +=fbuf[fpos].aB;
      }
    }
  }

//device render function k_render1
__global__ void k_render1
  (cu_particle_splotch *p, int nP,
  void *buf, bool a_eq_e, float grayabsorb,
  cu_exptable_info d_exp_info)
  {
  //first get the index m of this thread
  int m;
  m =blockIdx.x *blockDim.x + threadIdx.x;
  if (m >=nP)//m goes from 0 to nP-1
    return;

  //make fbuf the right type
  cu_fragment_AeqE        *fbuf;
  cu_fragment_AneqE       *fbuf1;
  if (a_eq_e)
    fbuf =(cu_fragment_AeqE*) buf;
  else
    fbuf1 =(cu_fragment_AneqE*)buf;

  //now do the calc
  const float powtmp = pow(Pi,1./3.);
  const float sigma0 = powtmp/sqrt(2*Pi);

  const float r = p[m].r;
  const float radsq = 2.25*r*r;
  const float stp = -0.5/(r*r*sigma0*sigma0);

  cu_color e=p[m].e, q;
  if (!a_eq_e)
   {
     q.r = e.r/(e.r+grayabsorb);
     q.g = e.g/(e.g+grayabsorb);
     q.b = e.b/(e.b+grayabsorb);
   }
  const float intens = -0.5/(2*sqrt(Pi)*powtmp);
  e.r*=intens; e.g*=intens; e.b*=intens;

  const float posx=p[m].x, posy=p[m].y;
  unsigned int fpos =p[m].posInFragBuf;

  if (a_eq_e)
  {
    for (int x=p[m].minx; x<p[m].maxx; ++x)
    {
     float dxsq=(x-posx)*(x-posx);
     for (int y=p[m].miny; y<p[m].maxy; ++y)
      {
        float dsq = (y-posy)*(y-posy) + dxsq;
        if (dsq<radsq)
        {
          float att = get_exp(stp*dsq, d_exp_info);
          fbuf[fpos].aR = att*e.r;
          fbuf[fpos].aG = att*e.g;
          fbuf[fpos].aB = att*e.b;
        }
        else
        {
          fbuf[fpos].aR =0.0;
          fbuf[fpos].aG =0.0;
          fbuf[fpos].aB =0.0;
        }
      //for each (x,y)
      fpos++;
      }//y
    }//x
  }
  else
  {
    for (int x=p[m].minx; x<p[m].maxx; ++x)
    {
     float dxsq=(x-posx)*(x-posx);
     for (int y=p[m].miny; y<p[m].maxy; ++y)
      {
        float dsq = (y-posy)*(y-posy) + dxsq;
        if (dsq<radsq)
        {
          float att = get_exp(stp*dsq, d_exp_info);
          float   expm1;
          expm1 =get_expm1(att*e.r, d_exp_info);
          fbuf1[fpos].aR = expm1;
          fbuf1[fpos].qR = q.r;
          expm1 =get_expm1(att*e.g, d_exp_info);
          fbuf1[fpos].aG = expm1;
          fbuf1[fpos].qG = q.g;
          expm1 =get_expm1(att*e.b, d_exp_info);
          fbuf1[fpos].aB = expm1;
          fbuf1[fpos].qB = q.b;
        }
        else
        {
          fbuf1[fpos].aR =0.0;
          fbuf1[fpos].aG =0.0;
          fbuf1[fpos].aB =0.0;
          fbuf1[fpos].qR =1.0;
          fbuf1[fpos].qG =1.0;
          fbuf1[fpos].qB =1.0;
        }
      //for each (x,y)
      fpos++;
      }//y
    }//x
  }
 }


//colorize by kernel
__global__ void k_colorize
  (cu_param_colorize *params, cu_particle_sim *p, int n, cu_particle_splotch *p2,
  cu_colormap_info info)
  {
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >n) m =n;

  //now do the calc, p[m]--->p2[m]
  p2[m].isValid=false;
  if (p[m].z<=0 || p[m].z<=params->zminval || p[m].z>=params->zmaxval)
    return;

  const float posx=p[m].x, posy=p[m].y;
  const float rfacr=params->rfac*p[m].r;

  // compute region occupied by the partile
  int minx=int(posx-rfacr+1);
  if (minx>=params->xres) return;
  minx=max(minx,0);

  int maxx=int(posx+rfacr+1);
  if (maxx<=0) return;
  maxx=min(maxx,params->xres);
  if (minx>=maxx) return;

  int miny=int(posy-rfacr+1);
  if (miny>=params->yres) return;
  miny=max(miny,0);

  int maxy=int(posy+rfacr+1);
  if (maxy<=0) return;
  maxy=min(maxy,params->yres);
  if (miny>=maxy) return;

  //set region info to output the p2
  p2[m].minx =minx;  p2[m].miny =miny;
  p2[m].maxx =maxx;  p2[m].maxy =maxy;

  float col1=p[m].e.r,col2=p[m].e.g,col3=p[m].e.b;
  clamp (0.0000001,0.9999999,col1);
  if (params->col_vector[p[m].type])
    {
    clamp (0.0000001,0.9999999,col2);
    clamp (0.0000001,0.9999999,col3);
    }
  float intensity=p[m].I;
  clamp (0.0000001,0.9999999,intensity);
  intensity *= params->brightness[p[m].type];

  cu_color e;
  if (params->col_vector[p[m].type])   // color from file
    {
    e.r=col1*intensity;
    e.g=col2*intensity;
    e.b=col3*intensity;
    }
  else   // get color, associated from physical quantity contained in e.r, from lookup table
    {
    e = get_color(p[m].type, col1, info);
    e.r *= intensity;
    e.g *= intensity;
    e.b *= intensity;
    }

  p2[m].isValid =true;
  p2[m].x =p[m].x;
  p2[m].y =p[m].y;
  p2[m].r =p[m].r;
  p2[m].e=e;
  }

//Range by kernel step 1
__global__ void k_range1(cu_param_range *pr, cu_particle_sim *p, int n)
  {
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >=n) m = n;

  //now do the calc
  //I, minint, maxint
  if (pr->log_int[p[m].type]) //could access invalid address under EMULATION
    p[m].I = log10(p[m].I);
  get_minmax(pr->minint[p[m].type], pr->maxint[p[m].type], p[m].I);

  //e.r, mincol, maxcol
  if (pr->log_col[p[m].type])
  p[m].e.r = log10(p[m].e.r);
  if (pr->asinh_col[p[m].type])
    p[m].e.r = my_asinh(p[m].e.r);
  get_minmax(pr->mincol[p[m].type], pr->maxcol[p[m].type], p[m].e.r);

  //C2, C3, mincol, maxcol
  if (pr->col_vector[p[m].type])
    {
    if (pr->log_col[p[m].type])
      {
      p[m].e.g = log10(p[m].e.g);
      p[m].e.b = log10(p[m].e.b);
      }
    if (pr->asinh_col[p[m].type])
      {
      p[m].e.g = my_asinh(p[m].e.g);
      p[m].e.b = my_asinh(p[m].e.b);
      }
    get_minmax(pr->mincol[p[m].type], pr->maxcol[p[m].type], p[m].e.g);
    get_minmax(pr->mincol[p[m].type], pr->maxcol[p[m].type], p[m].e.b);
    }
  }

//Range by kernel step 2
__global__ void k_range2
  (cu_param_range *pr, cu_particle_sim *p, int n, int itype,
  float minval_int, float maxval_int,
  float minval_col, float maxval_col)
  {
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >n) m =n;

  //do the calculation
  if(p[m].type == itype)///clamp into (min,max)
    {
    my_normalize(minval_int,maxval_int,p[m].I);
    my_normalize(minval_col,maxval_col,p[m].e.r);
    if (pr->col_vector[p[m].type])
      {
      my_normalize(minval_col,maxval_col,p[m].e.g);
      my_normalize(minval_col,maxval_col,p[m].e.b);
      }
    }
  }

//Transform by kernel
__global__ void k_transform
  (cu_particle_sim *p, int n, cu_param_transform *ptrans)
  {
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >n) m =n;

  //copy parameters to __share__ local memory? later

  //now do x,y,z
  float x,y,z;
  x =p[m].x*ptrans->p[0] + p[m].y*ptrans->p[1] + p[m].z*ptrans->p[2] + ptrans->p[3];
  y =p[m].x*ptrans->p[4] + p[m].y*ptrans->p[5] + p[m].z*ptrans->p[6] + ptrans->p[7];
  z =p[m].x*ptrans->p[8] + p[m].y*ptrans->p[9] + p[m].z*ptrans->p[10]+ ptrans->p[11];
  p[m].x =x;
  p[m].y =y;
  p[m].z =z;

  //do r
  float xfac = ptrans->xfac;
  const float   res2 = 0.5*ptrans->xres;
  const float   ycorr = .5f*(ptrans->yres-ptrans->xres);
  if (!ptrans->projection)
    {
    p[m].x = res2 * (p[m].x+ptrans->fovfct*ptrans->dist)*xfac;
    p[m].y = res2 * (p[m].y+ptrans->fovfct*ptrans->dist)*xfac + ycorr;
    }
  else
    {
    xfac=1./(ptrans->fovfct*p[m].z);
    p[m].x = res2 * (p[m].x+ptrans->fovfct*p[m].z)*xfac;
    p[m].y = res2 * (p[m].y+ptrans->fovfct*p[m].z)*xfac + ycorr;
    }

  p[m].I /= p[m].r;
  p[m].r = p[m].r *res2*xfac;

  const float rfac= sqrt(p[m].r*p[m].r + 0.25*ptrans->minrad_pix*ptrans->minrad_pix)/p[m].r;
  p[m].r *= rfac;
  p[m].I /= rfac;
  }


