// Portable CUDA/HIP empirical roofline benchmark. No fast-math; FMA counts as 2 FLOPs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>
using BF16=hip_bfloat16;
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaSetDevice hipSetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaSuccess hipSuccess
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
using BF16=__nv_bfloat16;
#endif
#define GPU(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
#define BLAS(x) do { int e=int(x); if(e) throw std::runtime_error(std::string(#x)+": status="+std::to_string(e)); } while(0)
__host__ __device__ float val(size_t i, unsigned seed) {
 unsigned x=unsigned(i)^seed; x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
 return (int(x%33)-16)/32.f;
}
template<class T> __global__ void init(T* p,size_t n,unsigned seed) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) p[i]=T(val(i,seed));
}
template<class T> struct Buf {
 T* p=nullptr; size_t n;
 Buf(size_t n_, unsigned seed=1):n(n_) {GPU(cudaMalloc((void**)&p,n*sizeof(T)));init<<<std::min(size_t(65535),(n+255)/256),256>>>(p,n,seed);GPU(cudaGetLastError());}
 ~Buf(){if(p) cudaFree(p);} Buf(const Buf&)=delete;
};
template<class T,int OP> __global__ void stream_kernel(const T* __restrict__ a,const T* __restrict__ b,T* __restrict__ c,size_t n) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
  if(OP==0)c[i]=a[i];
  if(OP==1)c[i]=a[i]+b[i];
  if(OP==2)c[i]=a[i]+T(2)*b[i];
 }
}
template<class T> __device__ T fused(T a,T b,T c) {if constexpr(std::is_same<T,float>::value)return fmaf(a,b,c);else return fma(a,b,c);}
// Four independent chains per thread; runtime coefficient prevents constant folding.
template<class T> __global__ void fma_kernel(const T* __restrict__ x,T* __restrict__ y,size_t n,int reps,T alpha,T beta) {
 for(size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;i<n;i+=size_t(gridDim.x)*blockDim.x*4) {
  T a=x[i],b=x[i+1],c=x[i+2],d=x[i+3];
  for(int j=0;j<reps;++j){a=fused(a,alpha,beta);b=fused(b,alpha,beta);c=fused(c,alpha,beta);d=fused(d,alpha,beta);}
  y[i]=a;y[i+1]=b;y[i+2]=c;y[i+3]=d;
 }
}
std::ofstream out; int samples=7; bool quick=false; cudaDeviceProp prop;
struct Timing {std::vector<double> ms; int loops;};
Timing timed(const std::function<void()>& fn, const std::function<void()>& prepare={}) {
 for(int i=0;i<5;++i) fn(); GPU(cudaDeviceSynchronize());
 cudaEvent_t a,b;GPU(cudaEventCreate(&a));GPU(cudaEventCreate(&b));
 GPU(cudaEventRecord(a));fn();GPU(cudaEventRecord(b));GPU(cudaEventSynchronize(b));float pilot;GPU(cudaEventElapsedTime(&pilot,a,b));
 int loops=prepare?1:std::max(1,std::min(200,int(20.0/std::max(.001f,pilot))));
 Timing t; t.loops=loops;
 for(int s=0;s<samples;++s){
  if(prepare){prepare();GPU(cudaDeviceSynchronize());}
  GPU(cudaEventRecord(a));for(int j=0;j<loops;++j)fn();GPU(cudaEventRecord(b));GPU(cudaEventSynchronize(b));
  float ms;GPU(cudaEventElapsedTime(&ms,a,b));t.ms.push_back(ms/loops);
 }
 GPU(cudaEventDestroy(a));GPU(cudaEventDestroy(b));GPU(cudaGetLastError());return t;
}
void emit(std::string family,std::string dtype,std::string cache,int m,int n,int k,int reps,double bytes,double flops,size_t footprint,const Timing&t,double err,bool ok) {
 for(size_t s=0;s<t.ms.size();++s) out<<family<<','<<dtype<<','<<cache<<','<<m<<','<<n<<','<<k<<','<<reps<<','<<bytes<<','<<flops<<','<<footprint<<','<<s<<','<<t.loops<<','<<t.ms[s]<<','<<err<<','<<int(ok)<<'\n';
 out.flush();std::cout<<family<<' '<<dtype<<' '<<cache<<' '<<m<<'x'<<n<<'x'<<k<<" err="<<err<<" ok="<<ok<<std::endl;
 if(!ok) throw std::runtime_error("Numerical validation failed");
}
template<class T> double get(T* p,size_t i){T x;GPU(cudaMemcpy(&x,p+i,sizeof(T),cudaMemcpyDeviceToHost));double v=double(x);if(!std::isfinite(v))throw std::runtime_error("Nonfinite GPU output");return v;}
template<class T,int OP> void memory_case(size_t n,std::string dtype) {
 Buf<T>a(n,17),b(n,29),c(n,43);
 auto fn=[&](){stream_kernel<T,OP><<<std::min(size_t(65535),(n+255)/256),256>>>(a.p,b.p,c.p,n);};
 auto t=timed(fn);double err=0;
 for(int j=0;j<64;++j){size_t i=(size_t(j)*104729)%n;double expected=val(i,17)+(OP==0?0:val(i,29)*(OP==2?2:1));err=std::max(err,std::abs(get(c.p,i)-expected));}
 double bytes=double(n)*sizeof(T)*(OP==0?2:3);
 emit(OP==0?"copy":OP==1?"add":"triad",dtype,"warm",0,int(n),0,0,bytes,double(n)*OP,size_t(bytes),t,err,err<1e-6);
}
template<class T> void fma_case(int reps,std::string dtype) {
 size_t n=(size_t(1)<<28)/sizeof(T); Buf<T>x(n,17),y(n,23);
 T alpha=T(.9990234375),beta=T(.0009765625);
 auto fn=[&](){fma_kernel<T><<<8192,256>>>(x.p,y.p,n,reps,alpha,beta);}; auto t=timed(fn);double err=0;
 for(int j=0;j<32;++j){size_t i=(size_t(j)*104729)%n;T z=T(val(i,17));for(int r=0;r<reps;++r)z=std::fma(z,alpha,beta);err=std::max(err,std::abs(get(y.p,i)-double(z)));}
 emit("fma",dtype,"streaming",0,int(n),0,reps,double(n)*sizeof(T)*2,2.0*n*reps,n*sizeof(T)*2,t,err,err<1e-6);
}
#ifdef USE_HIP
rocblas_handle handle;
template<class T> rocblas_datatype datatype();
template<> rocblas_datatype datatype<float>(){return rocblas_datatype_f32_r;}
template<> rocblas_datatype datatype<double>(){return rocblas_datatype_f64_r;}
template<> rocblas_datatype datatype<__half>(){return rocblas_datatype_f16_r;}
template<> rocblas_datatype datatype<BF16>(){return rocblas_datatype_bf16_r;}
template<class T> void gemm(int m,int n,int k,T*a,T*b,T*c,bool tf32) {
 using Acc=typename std::conditional<std::is_same<T,double>::value,double,float>::type;Acc one=1,zero=0;
 BLAS(rocblas_gemm_ex(handle,rocblas_operation_none,rocblas_operation_none,m,n,k,&one,a,datatype<T>(),m,b,datatype<T>(),k,&zero,c,datatype<T>(),m,c,datatype<T>(),m,datatype<Acc>(),rocblas_gemm_algo_standard,0,0));
}
#else
cublasHandle_t handle;
template<class T> cudaDataType datatype();
template<> cudaDataType datatype<float>(){return CUDA_R_32F;}
template<> cudaDataType datatype<double>(){return CUDA_R_64F;}
template<> cudaDataType datatype<__half>(){return CUDA_R_16F;}
template<> cudaDataType datatype<BF16>(){return CUDA_R_16BF;}
template<class T> void gemm(int m,int n,int k,T*a,T*b,T*c,bool tf32) {
 using Acc=typename std::conditional<std::is_same<T,double>::value,double,float>::type;Acc one=1,zero=0;
 auto compute=std::is_same<T,double>::value?CUBLAS_COMPUTE_64F:tf32?CUBLAS_COMPUTE_32F_FAST_TF32:std::is_same<T,float>::value?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F;
 BLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,a,datatype<T>(),m,b,datatype<T>(),k,&zero,c,datatype<T>(),m,compute,CUBLAS_GEMM_DEFAULT));
}
#endif
Buf<float>*flushbuf=nullptr;
void flush_cache(){init<<<8192,256>>>(flushbuf->p,flushbuf->n,193);}
template<class T> void gemm_case(int m,int n,int k,std::string dtype,bool cold=false,bool tf32=false) {
 Buf<T>a(size_t(m)*k,17),b(size_t(k)*n,29),c(size_t(m)*n,43);
 auto fn=[&](){gemm(m,n,k,a.p,b.p,c.p,tf32);};auto t=timed(fn,cold?std::function<void()>(flush_cache):std::function<void()>());
 double err=0,scale=0;
 for(int j=0;j<64;++j){int row=(j*104729+7)%m,col=(j*7919+3)%n;double ref=0;for(int kk=0;kk<k;++kk)ref+=double(val(row+size_t(kk)*m,17))*val(kk+size_t(col)*k,29);
  // Normalize by reference RMS below; do not hide near-zero errors with per-element relative error.
  double diff=get(c.p,row+size_t(col)*m)-ref;err+=diff*diff;scale+=ref*ref;
 }
 err=std::sqrt(err/std::max(scale,1e-30));double tol=dtype=="bf16"?.008:dtype=="fp16"?.001:dtype=="tf32"?.002:dtype=="fp64"?1e-12:1e-5;
 size_t bytes=(size_t(m)*k+size_t(k)*n+size_t(m)*n)*sizeof(T);
 emit("gemm",dtype,cold?"cold":"warm",m,n,k,0,bytes,2.0*m*n*k,bytes,t,err,std::isfinite(err)&&err<tol);
}
template<class T> void gemm_suite(std::string dtype,bool tf32=false){
 for(int n: {256,512,1024,2048,4096,8192}){if(quick&&n!=1024)continue;gemm_case<T>(n,n,n,dtype,false,tf32);if(n<=2048)gemm_case<T>(n,n,n,dtype,true,tf32);}
 if(!quick)for(int m:{1,2,4,8,16,32,64,128,256,512,1024,2048})gemm_case<T>(m,8192,8192,dtype,false,tf32);
}
int main(int argc,char**argv){try{
 if(argc<2)throw std::runtime_error("usage: bench OUTPUT_DIR [quick]");std::string dir=argv[1];quick=argc>2&&std::string(argv[2])=="quick";samples=quick?3:7;
 int count;GPU(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("Expected exactly one allocated visible GPU, got "+std::to_string(count));GPU(cudaSetDevice(0));GPU(cudaGetDeviceProperties(&prop,0));
 std::ofstream meta(dir+"/device.json");meta<<"{\"name\":\""<<prop.name<<"\",\"memory_bytes\":"<<prop.totalGlobalMem<<",\"l2_bytes\":"<<prop.l2CacheSize<<",\"sm_count\":"<<prop.multiProcessorCount<<",\"major\":"<<prop.major<<",\"minor\":"<<prop.minor<<",\"backend\":\""
#ifdef USE_HIP
 <<"hip\",\"arch\":\""<<prop.gcnArchName
#else
 <<"cuda\",\"arch\":\"sm_"<<prop.major<<prop.minor
#endif
 <<"\",\"quick\":"<<(quick?"true":"false")<<"}\n";meta.close();
#ifdef USE_HIP
 BLAS(rocblas_create_handle(&handle));
#else
 BLAS(cublasCreate(&handle));BLAS(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
#endif
 out.open(dir+"/samples.csv");out<<std::setprecision(15)<<"family,dtype,cache,m,n,k,reps,bytes,flops,footprint_bytes,sample,loops,ms,error,valid\n";
 for(int power:{12,16,20,22,24,26,28}){if(quick&&power!=20)continue;size_t n=(size_t(1)<<power)/4;memory_case<float,0>(n,"fp32");memory_case<float,1>(n,"fp32");memory_case<float,2>(n,"fp32");}
 for(int reps:{1,2,4,8,16,32,64,128,256,512,1024,2048}){if(quick&&reps!=4)continue;fma_case<float>(reps,"fp32");fma_case<double>(reps,"fp64");}
 flushbuf=new Buf<float>(std::max(size_t(1)<<27,size_t(prop.l2CacheSize)*4)/4,13);
 gemm_suite<float>("fp32");gemm_suite<double>("fp64");gemm_suite<__half>("fp16");
#ifdef USE_HIP
 gemm_suite<BF16>("bf16");
#else
 if(prop.major>=8){gemm_suite<BF16>("bf16");gemm_suite<float>("tf32",true);}
#endif
 delete flushbuf;
#ifdef USE_HIP
 BLAS(rocblas_destroy_handle(handle));
#else
 BLAS(cublasDestroy(handle));
#endif
 std::ofstream(dir+"/COMPLETE")<<"All cases completed and sampled numerical checks passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FATAL: "<<e.what()<<std::endl;return 1;}}
