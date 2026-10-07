#include <cuda_runtime.h>
#include <cstdio>
int main(){
 int count=0;auto e=cudaGetDeviceCount(&count);
 if(e!=cudaSuccess||count!=1){std::fprintf(stderr,"Expected one allocated CUDA device: %s, count=%d\n",cudaGetErrorString(e),count);return 1;}
 cudaDeviceProp p{};e=cudaGetDeviceProperties(&p,0);
 if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}
 char pci[64]{};e=cudaDeviceGetPCIBusId(pci,sizeof(pci),0);if(e!=cudaSuccess)return 1;
 std::printf("{\"name\":\"%s\",\"memory_bytes\":%zu,\"sm_count\":%d,\"major\":%d,\"minor\":%d,\"pci_bus_id\":\"%s\",\"uuid\":\"GPU-",p.name,p.totalGlobalMem,p.multiProcessorCount,p.major,p.minor,pci);
 for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)std::printf("-");std::printf("%02x",static_cast<unsigned char>(p.uuid.bytes[i]));}
 std::printf("\"}\n");return 0;
}
