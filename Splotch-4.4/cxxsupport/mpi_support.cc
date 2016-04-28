#ifdef USE_MPI
#include "mpi.h"
#else
#include <cstring>
#include <cstdlib>
#endif
#include "mpi_support.h"

MPI_Manager mpiMgr;

using namespace std;

#ifdef USE_MPI

namespace {

MPI_Datatype ndt2mpi (NDT type)
  {
  switch (type)
    {
    case NAT_CHAR: return MPI_CHAR;
    case NAT_INT: return MPI_INT;
    case NAT_UINT: return MPI_UNSIGNED;
    case NAT_LONG: return MPI_LONG;
    case NAT_ULONG: return MPI_UNSIGNED_LONG;
    case NAT_LONGLONG: return MPI_LONG_LONG;
    case NAT_ULONGLONG: return MPI_UNSIGNED_LONG_LONG;
    case NAT_FLOAT: return MPI_FLOAT;
    case NAT_DOUBLE: return MPI_DOUBLE;
    case NAT_LONGDOUBLE: return MPI_LONG_DOUBLE;
    default: planck_fail ("Unsupported type");
    }
  }
MPI_Op op2mop (MPI_Manager::redOp op)
  {
  switch (op)
    {
    case MPI_Manager::Min: return MPI_MIN;
    case MPI_Manager::Max: return MPI_MAX;
    case MPI_Manager::Sum: return MPI_SUM;
    default: planck_fail ("unsupported reduction operation");
    }
  }

} // unnamed namespace

#endif

#ifdef USE_MPI

void MPI_Manager::gatherv_helper1_m (int nval_loc, arr<int> &nval,
  arr<int> &offset, int &nval_tot) const
  {
  gather_m (nval_loc, nval);
  nval_tot=0;
  for (tsize i=0; i<nval.size(); ++i)
    nval_tot+=nval[i];
  offset.alloc(num_ranks());
  offset[0]=0;
  for (tsize i=1; i<offset.size(); ++i)
    offset[i]=offset[i-1]+nval[i-1];
  }

#else

void MPI_Manager::gatherv_helper1_m (int nval_loc, arr<int> &nval,
  arr<int> &offset, int &nval_tot) const
  {
  nval.alloc(1);
  nval[0]=nval_tot=nval_loc;
  offset.alloc(1);
  offset[0]=0;
  }

#endif

#ifdef USE_MPI

MPI_Manager::MPI_Manager ()
  {
  MPI_Init(0,0);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
  }
MPI_Manager::~MPI_Manager ()
  { MPI_Finalize(); }

void MPI_Manager::abort() const
  { MPI_Abort(MPI_COMM_WORLD, 1); }

int MPI_Manager::num_ranks() const
  { int res; MPI_Comm_size(MPI_COMM_WORLD, &res); return res; }
int MPI_Manager::rank() const
  { int res; MPI_Comm_rank(MPI_COMM_WORLD, &res); return res; }
bool MPI_Manager::master() const
  { return (rank() == 0); }

void MPI_Manager::barrier() const
  { MPI_Barrier(MPI_COMM_WORLD); }

#else

MPI_Manager::MPI_Manager () {}
MPI_Manager::~MPI_Manager () {}

void MPI_Manager::abort() const
  { exit(1); }

int MPI_Manager::num_ranks() const { return 1; }
int MPI_Manager::rank() const { return 0; }
bool MPI_Manager::master() const { return true; }

void MPI_Manager::barrier() const {}

#endif

#ifdef USE_MPI
void MPI_Manager::sendRawVoid (const void *data, NDT type, tsize num,
  tsize dest) const
  {
  MPI_Send(const_cast<void *>(data),num,ndt2mpi(type),dest,0,MPI_COMM_WORLD);
  }
void MPI_Manager::recvRawVoid (void *data, NDT type, tsize num, tsize src) const
  { MPI_Recv(data,num,ndt2mpi(type),src,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
void MPI_Manager::sendrecvRawVoid (const void *sendbuf, tsize sendcnt,
  tsize dest, void *recvbuf, tsize recvcnt, tsize src, NDT type) const
  {
  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Sendrecv (const_cast<void *>(sendbuf),sendcnt,dtype,dest,0,
    recvbuf,recvcnt,dtype,src,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
void MPI_Manager::sendrecv_replaceRawVoid (void *data, NDT type, tsize num,
  tsize dest, tsize src) const
  {
  MPI_Sendrecv_replace (data,num,ndt2mpi(type),dest,0,src,0,MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);
  }

void MPI_Manager::gatherRawVoid (const void *in, tsize num, void *out, NDT type)
  const
  {
  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Gather(const_cast<void *>(in),1,dtype,out,num,dtype,0,MPI_COMM_WORLD);
  }
void MPI_Manager::gathervRawVoid (const void *in, tsize num, void *out,
  const int *nval, const int *offset, NDT type) const
  {
  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Gatherv(const_cast<void *>(in),num,dtype,out,const_cast<int *>(nval),
    const_cast<int *>(offset),dtype,0,MPI_COMM_WORLD);
  }

void MPI_Manager::allgatherRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  {
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Allgather (const_cast<void *>(in),num,tp,out,num,tp,MPI_COMM_WORLD);
  }
void MPI_Manager::allreduceRawVoid (void *data, NDT type,
  tsize num, redOp op) const
  {
  MPI_Allreduce (MPI_IN_PLACE,data,num,ndt2mpi(type),op2mop(op),
    MPI_COMM_WORLD);
  }
void MPI_Manager::allreduceRawVoid (const void *in, void *out, NDT type,
  tsize num, redOp op) const
  {
  MPI_Allreduce (const_cast<void *>(in),out,num,ndt2mpi(type),op2mop(op),
    MPI_COMM_WORLD);
  }
void MPI_Manager::reduceRawVoid (const void *in, void *out, NDT type, tsize num,
  redOp op, int root) const
  {
  MPI_Reduce (const_cast<void *>(in),out,num,ndt2mpi(type),op2mop(op), root,
    MPI_COMM_WORLD);
  }

void MPI_Manager::bcastRawVoid (void *data, NDT type, tsize num, int root) const
  { MPI_Bcast (data,num,ndt2mpi(type),root,MPI_COMM_WORLD); }

void MPI_Manager::all2allRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  {
  tsize nranks = num_ranks();
  planck_assert (num%nranks==0,
    "array size is not divisible by number of ranks");
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Alltoall (const_cast<void *>(in),num/nranks,tp,out,num/nranks,tp,
    MPI_COMM_WORLD);
  }

void MPI_Manager::all2allvRawVoid (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, NDT type)
  const
  {
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Alltoallv (const_cast<void *>(in), const_cast<int *>(numin),
    const_cast<int *>(disin), tp, out, const_cast<int *>(numout),
    const_cast<int *>(disout), tp, MPI_COMM_WORLD);
  }

#else

void MPI_Manager::sendRawVoid (const void *, NDT, tsize, tsize) const
  { planck_fail("not supported in scalar code"); }
void MPI_Manager::recvRawVoid (void *, NDT, tsize, tsize) const
  { planck_fail("not supported in scalar code"); }
void MPI_Manager::sendrecvRawVoid (const void *, tsize, tsize, void *, tsize,
  tsize, NDT) const
  { planck_fail("not supported in scalar code"); }
void MPI_Manager::sendrecv_replaceRawVoid (void *, NDT, tsize, tsize dest,
  tsize src) const
  { planck_assert ((dest==0) && (src==0), "inconsistent call"); }

void MPI_Manager::gatherRawVoid (const void *in, tsize num, void *out, NDT type)
  const
  { memcpy (out, in, num*ndt2size(type)); }
void MPI_Manager::gathervRawVoid (const void *in, tsize num, void *out,
  const int *, const int *, NDT type) const
  { memcpy (out, in, num*ndt2size(type)); }
void MPI_Manager::allgatherRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  { memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::allreduceRawVoid (const void *in, void *out, NDT type,
  tsize num, redOp) const
  { memcpy (out, in, num*ndt2size(type)); }
void MPI_Manager::allreduceRawVoid (void *, NDT, tsize, redOp) const
  {}
void MPI_Manager::reduceRawVoid (const void *in, void *out, NDT type, tsize num,
  redOp, int) const
  { memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::bcastRawVoid (void *, NDT, tsize, int) const
  {}

void MPI_Manager::all2allRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  { memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::all2allvRawVoid (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, NDT type)
  const
  {
  planck_assert (numin[0]==numout[0],"message size mismatch");
  const char *in2 = static_cast<const char *>(in);
  char *out2 = static_cast<char *>(out);
  tsize st=ndt2size(type);
  memcpy (out2+disout[0]*st,in2+disin[0]*st,numin[0]*st);
  }

#endif
