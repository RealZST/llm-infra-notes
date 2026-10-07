// Mostly-read bandwidth calibration: one 4 MiB partial-sum output per launch.
#define main benchmark_main
#include "bench.cu"
#undef main
__global__ void read_reduce(const float* __restrict__ x,float* __restrict__ y,size_t n){
 size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x,stride=size_t(gridDim.x)*blockDim.x;
 float a=0,b=0,c=0,d=0;
 for(size_t i=tid;i<n;i+=stride*4){a+=x[i];b+=x[i+stride];c+=x[i+stride*2];d+=x[i+stride*3];}
 y[tid]=(a+b)+(c+d);
}
int main(int argc,char**argv){try{
 if(argc<2)throw std::runtime_error("usage: bandwidth OUTPUT_DIR");std::string dir=argv[1];GPU(cudaSetDevice(0));GPU(cudaGetDeviceProperties(&prop,0));
 out.open(dir+"/samples.csv");out<<std::setprecision(15)<<"family,dtype,cache,m,n,k,reps,bytes,flops,footprint_bytes,sample,loops,ms,error,valid\n";
 const size_t threads=4096*256;
 for(int p:{26,28,29}){size_t n=(size_t(1)<<p)/4;Buf<float>x(n,17),y(threads,43);
  auto fn=[&](){read_reduce<<<4096,256>>>(x.p,y.p,n);};auto t=timed(fn);double err=0;
  for(int j=0;j<64;++j){size_t tid=(size_t(j)*104729)%threads;double ref=0;for(size_t i=tid;i<n;i+=threads)ref+=val(i,17);err=std::max(err,std::abs(get(y.p,tid)-ref));}
  // FLOPs left at 0: calibration-only record, not a roofline workload point.
  emit("read_reduce","fp32","warm",0,int(n),0,0,(n+threads)*4,0,(n+threads)*4,t,err,err<1e-6);
 }
 std::ofstream(dir+"/COMPLETE")<<"Read reduction outputs checked against CPU reference\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
