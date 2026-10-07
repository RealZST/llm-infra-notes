// Native cuBLASLt E4M3, FP32 computation, FP16 output, unit scales; no sparsity.
#define main benchmark_main
#include "bench.cu"
#undef main
#include <cuda_fp8.h>
#include <cublasLt.h>
void fp8_case(cublasLtHandle_t lt,void*workspace,int m,int n,int k){
 Buf<__nv_fp8_e4m3>a(size_t(m)*k,17),b(size_t(k)*n,29);Buf<__half>c(size_t(m)*n,43);Buf<float>scale(1);float one=1,zero=0;GPU(cudaMemcpy(scale.p,&one,4,cudaMemcpyHostToDevice));
 cublasLtMatmulDesc_t desc;BLAS(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scale.p,sizeof(scale.p)));
 BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scale.p,sizeof(scale.p)));
 int8_t fast=0;BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast)));
 cublasLtMatrixLayout_t al,bl,cl;
 BLAS(cublasLtMatrixLayoutCreate(&al,CUDA_R_8F_E4M3,k,m,k));BLAS(cublasLtMatrixLayoutCreate(&bl,CUDA_R_8F_E4M3,k,n,k));BLAS(cublasLtMatrixLayoutCreate(&cl,CUDA_R_16F,m,n,m));
 cublasLtMatmulPreference_t pref;BLAS(cublasLtMatmulPreferenceCreate(&pref));size_t ws=64<<20;
 BLAS(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
 cublasLtMatmulHeuristicResult_t algo;int count;
 BLAS(cublasLtMatmulAlgoGetHeuristic(lt,desc,al,bl,cl,cl,pref,1,&algo,&count));if(!count)throw std::runtime_error("No FP8 heuristic for "+std::to_string(m));
 auto fn=[&](){BLAS(cublasLtMatmul(lt,desc,&one,a.p,al,b.p,bl,&zero,c.p,cl,c.p,cl,&algo.algo,workspace,ws,0));};auto t=timed(fn);
 double err=0,scale2=0;
 for(int j=0;j<64;++j){int row=(j*104729+7)%m,col=(j*7919+3)%n;double ref=0;for(int kk=0;kk<k;++kk)ref+=double(val(kk+size_t(row)*k,17))*val(kk+size_t(col)*k,29);double diff=get(c.p,row+size_t(col)*m)-ref;err+=diff*diff;scale2+=ref*ref;}
 err=std::sqrt(err/std::max(scale2,1e-30));size_t bytes=size_t(m)*k+size_t(k)*n+size_t(m)*n*2+8;
 emit("gemm","fp8_e4m3","warm",m,n,k,0,bytes,2.*m*n*k,bytes,t,err,std::isfinite(err)&&err<.001);
 BLAS(cublasLtMatmulPreferenceDestroy(pref));BLAS(cublasLtMatrixLayoutDestroy(al));BLAS(cublasLtMatrixLayoutDestroy(bl));BLAS(cublasLtMatrixLayoutDestroy(cl));BLAS(cublasLtMatmulDescDestroy(desc));
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: fp8 OUTPUT_DIR");std::string dir=argv[1];GPU(cudaSetDevice(0));GPU(cudaGetDeviceProperties(&prop,0));
 out.open(dir+"/samples.csv");out<<std::setprecision(15)<<"family,dtype,cache,m,n,k,reps,bytes,flops,footprint_bytes,sample,loops,ms,error,valid\n";
 cublasLtHandle_t lt;BLAS(cublasLtCreate(&lt));void*workspace;GPU(cudaMalloc(&workspace,64<<20));
 for(int n:{256,512,1024,2048,4096,8192})fp8_case(lt,workspace,n,n,n);
 for(int m:{16,32,64,128,256,512,1024,2048})fp8_case(lt,workspace,m,8192,8192);
 GPU(cudaFree(workspace));BLAS(cublasLtDestroy(lt));std::ofstream(dir+"/COMPLETE")<<"FP8 checked against CPU FP64 reference\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
