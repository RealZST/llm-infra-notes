// Native NVFP4 E2M1 with varying per-16-element E4M3 scales; BF16 output.
// Scale descriptor API checked against NVIDIA CUDALibrarySamples (Apache-2.0).
#define main benchmark_main
#include "bench.cu"
#undef main
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublasLt.h>
__host__ __device__ unsigned code4(size_t i,unsigned seed){unsigned x=unsigned(i)^seed;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x&15;}
__host__ __device__ float block_scale(int row,int kb){return float(1<<((row*3+kb*7)%4))/32.f;}
float dequant4(size_t i,unsigned seed,int k){static const float mag[]={0,.5,1,1.5,2,3,4,6};unsigned c=code4(i,seed);return (c&8?-1.f:1.f)*mag[c&7]*block_scale(i/k,(i%k)/16);}
__global__ void pack4(unsigned char*p,size_t bytes,unsigned seed){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bytes;i+=size_t(gridDim.x)*blockDim.x)p[i]=code4(i*2,seed)|(code4(i*2+1,seed)<<4);}
size_t scale_size(int rows,int k){return size_t((rows+127)/128)*128*((k/16+3)/4)*4;}
void init_scales(__nv_fp8_e4m3*p,int rows,int k){
 int pr=(rows+127)/128*128,pk=(k/16+3)/4*4;std::vector<__nv_fp8_e4m3> host(size_t(pr)*pk);
 for(int r=0;r<pr;++r)for(int c=0;c<pk;++c){size_t offset=(size_t(r/128)*(pk/4)+c/4)*512+(r%32)*16+((r%128)/32)*4+c%4;host[offset]=__nv_fp8_e4m3(r<rows && c<k/16 ? block_scale(r,c) : 0.f);}
 GPU(cudaMemcpy(p,host.data(),host.size(),cudaMemcpyHostToDevice));
}
void fp8_case(cublasLtHandle_t lt,void*workspace,int m,int n,int k){
 Buf<unsigned char>a(size_t(m)*k/2),b(size_t(k)*n/2);Buf<BF16>c(size_t(m)*n,43);Buf<__nv_fp8_e4m3>sa(scale_size(m,k)),sb(scale_size(n,k));float one=1,zero=0;
 pack4<<<8192,256>>>(a.p,a.n,17);pack4<<<8192,256>>>(b.p,b.n,29);init_scales(sa.p,m,k);init_scales(sb.p,n,k);
 cublasLtMatmulDesc_t desc;BLAS(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 auto smode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&smode,sizeof(smode)));
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&smode,sizeof(smode)));
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa.p,sizeof(sa.p)));
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb.p,sizeof(sb.p)));
 int8_t fast=0;BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast)));
 cublasLtMatrixLayout_t al,bl,cl;
 BLAS(cublasLtMatrixLayoutCreate(&al,CUDA_R_4F_E2M1,k,m,k));BLAS(cublasLtMatrixLayoutCreate(&bl,CUDA_R_4F_E2M1,k,n,k));BLAS(cublasLtMatrixLayoutCreate(&cl,CUDA_R_16BF,m,n,m));
 cublasLtMatmulPreference_t pref;BLAS(cublasLtMatmulPreferenceCreate(&pref));size_t ws=64<<20;
 BLAS(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
 cublasLtMatmulHeuristicResult_t algo;int count;
 BLAS(cublasLtMatmulAlgoGetHeuristic(lt,desc,al,bl,cl,cl,pref,1,&algo,&count));if(!count)throw std::runtime_error("No NVFP4 heuristic for "+std::to_string(m));
 auto fn=[&](){BLAS(cublasLtMatmul(lt,desc,&one,a.p,al,b.p,bl,&zero,c.p,cl,c.p,cl,&algo.algo,workspace,ws,0));};auto t=timed(fn);
 double err=0,scale2=0;
 for(int j=0;j<64;++j){int row=(j*104729+7)%m,col=(j*7919+3)%n;double ref=0;for(int kk=0;kk<k;++kk)ref+=double(dequant4(kk+size_t(row)*k,17,k))*dequant4(kk+size_t(col)*k,29,k);double diff=get(c.p,row+size_t(col)*m)-ref;err+=diff*diff;scale2+=ref*ref;}
 err=std::sqrt(err/std::max(scale2,1e-30));size_t bytes=(size_t(m)*k+size_t(k)*n)/2+size_t(m)*n*2+sa.n+sb.n;
 emit("gemm","nvfp4","warm",m,n,k,0,bytes,2.*m*n*k,bytes,t,err,std::isfinite(err)&&err<.008);
 BLAS(cublasLtMatmulPreferenceDestroy(pref));BLAS(cublasLtMatrixLayoutDestroy(al));BLAS(cublasLtMatrixLayoutDestroy(bl));BLAS(cublasLtMatrixLayoutDestroy(cl));BLAS(cublasLtMatmulDescDestroy(desc));
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: fp4 OUTPUT_DIR");std::string dir=argv[1];GPU(cudaSetDevice(0));GPU(cudaGetDeviceProperties(&prop,0));
 out.open(dir+"/samples.csv");out<<std::setprecision(15)<<"family,dtype,cache,m,n,k,reps,bytes,flops,footprint_bytes,sample,loops,ms,error,valid\n";
 cublasLtHandle_t lt;BLAS(cublasLtCreate(&lt));void*workspace;GPU(cudaMalloc(&workspace,64<<20));
 for(int n:{256,512,1024,2048,4096,8192})fp8_case(lt,workspace,n,n,n);
 for(int m:{16,32,64,128,256,512,1024,2048})fp8_case(lt,workspace,m,8192,8192);
 GPU(cudaFree(workspace));BLAS(cublasLtDestroy(lt));std::ofstream(dir+"/COMPLETE")<<"NVFP4 checked against CPU FP64 reference\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
