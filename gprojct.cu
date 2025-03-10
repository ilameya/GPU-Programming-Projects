#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>



// data for the real galaxies will be read into these arrays
float *real_rasc, *real_decl;
// data for the simulated random galaxies will be read into these arrays
float *rand_rasc, *rand_decl;

unsigned long long int *histogramDR, *histogramDD, *histogramRR;
float   pif = acosf(-1.0f); // PI 
long int CPUMemory = 0L;
long int GPUMemory = 0L;

int totaldegrees = 360;
int binsperdegree = 4;

long int NUM_OF_GALAXIES = 100000L;

// GPU kernel(s) to calculate the histograms
__global__ void fillHistogram(unsigned long long* hist, float* r1_rasc, float* r1_decl, float* r2_rasc, float* r2_decl, long int N)
{
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        int j;
        double angle; // angle between galaxies
        float rad_2_deg = 180/acosf(-1.0f);   // radians to degrees
        for(j=0;j<N;j++)
        {

           angle = sinf(r1_decl[tid])*sinf(r2_decl[j]) + cosf(r1_decl[tid])*cosf(r2_decl[j])*cosf(r1_rasc[tid]-r2_rasc[j]);
           if ( angle > 1.0f ) angle = 1.0f;
           if (angle < -1.0f ) angle = -1.0f;
           angle = rad_2_deg*acosf(angle);
            atomicAdd(&hist[ (int) floor(angle / 0.25) ],1L);
        }
    }
}

int main(int argc, char *argv[])
{
   int parseargs_readinput(int argc, char *argv[]);
   long int histogramDRsum, histogramDDsum, histogramRRsum;
   double walltime;
   struct timeval _ttime;
   struct timezone _tzone;
   int getDevice(void);
   FILE *output_file;
    
   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   gettimeofday(&_ttime, &_tzone);
   walltime = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    
    // Right ascension and declination for real galaxies 
   real_rasc        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));
   real_decl        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));

    // Right ascension and declination for synthetic random galaxies 
   rand_rasc        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));
   rand_decl        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));
   CPUMemory += 4L*NUM_OF_GALAXIES*sizeof(float);
    
    if ( parseargs_readinput(argc, argv) != 0 ) {printf("   Program stopped.\n");return(0);}
    printf("   Input data read, now calculating histograms\n");

   if ( getDevice() != 0 ) return(-1);

   // allocate memory on CPU for the histograms DD, DR, and RR
   histogramDR = (unsigned long long int *)calloc(totaldegrees*binsperdegree+1L,sizeof(unsigned long long int));
   histogramDD = (unsigned long long int *)calloc(totaldegrees*binsperdegree+1L,sizeof(unsigned long long int));
   histogramRR = (unsigned long long int *)calloc(totaldegrees*binsperdegree+1L,sizeof(unsigned long long int));
   CPUMemory += 3L*(totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int);
   
   
   // allocate memory on the GPU for input data 
   float* d_real_rasc; cudaMalloc(&d_real_rasc, NUM_OF_GALAXIES*sizeof(float));
   float* d_real_decl; cudaMalloc(&d_real_decl, NUM_OF_GALAXIES*sizeof(float));
   float* d_rand_rasc; cudaMalloc(&d_rand_rasc, NUM_OF_GALAXIES*sizeof(float));
   float* d_rand_decl; cudaMalloc(&d_rand_decl, NUM_OF_GALAXIES*sizeof(float));
   GPUMemory += 4L*NUM_OF_GALAXIES*sizeof(float);
    
   // Allocate memory on the GPU for histograms
   unsigned long long* d_histogramDR; cudaMalloc(&d_histogramDR, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int));
   unsigned long long* d_histogramDD; cudaMalloc(&d_histogramDD, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int));
   unsigned long long* d_histogramRR; cudaMalloc(&d_histogramRR, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int));
   GPUMemory += 3L*(totaldegrees*binsperdegree+1L)*sizeof(long int);
    
   // Copy the input data to the GPU
   cudaMemcpy(d_real_rasc, real_rasc, NUM_OF_GALAXIES*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_real_decl, real_decl, NUM_OF_GALAXIES*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_rand_rasc, rand_rasc, NUM_OF_GALAXIES*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_rand_decl, rand_decl, NUM_OF_GALAXIES*sizeof(float), cudaMemcpyHostToDevice);
   
   // call the GPU kernel(s) to fill the histograms
   int block_size = 256;
   int grid_size = ((NUM_OF_GALAXIES + block_size) / block_size);
   fillHistogram<<<grid_size,block_size>>>(d_histogramDD, d_real_rasc, d_real_decl, d_real_rasc, d_real_decl, NUM_OF_GALAXIES);
   fillHistogram<<<grid_size,block_size>>>(d_histogramDR, d_real_rasc, d_real_decl, d_rand_rasc, d_rand_decl, NUM_OF_GALAXIES);
   fillHistogram<<<grid_size,block_size>>>(d_histogramRR, d_rand_rasc, d_rand_decl, d_rand_rasc, d_rand_decl, NUM_OF_GALAXIES);
    cudaDeviceSynchronize();

   // Copy data back to the CPU. (copying the histograms data)
   cudaMemcpy(histogramDR, d_histogramDR, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramDD, d_histogramDD, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramRR, d_histogramRR, (totaldegrees*binsperdegree+1L)*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

    

// Sanitry checks on the histograms
   histogramDRsum = 0L;
   for ( int i = 0; i < binsperdegree*totaldegrees;++i ) histogramDRsum += histogramDR[i];
   printf("   DR histogram sum = %ld\n",histogramDRsum);
   if ( histogramDRsum != 10000000000L ) {
      printf("   Incorrect histogram sum, exiting..\n");return(0);
   }

   histogramDDsum = 0L;
   for ( int i = 0; i < binsperdegree*totaldegrees;++i )
        histogramDDsum += histogramDD[i];
   printf("   DD histogram sum = %ld\n",histogramDDsum);
   if ( histogramDDsum != 10000000000L ) {
      printf("   Incorrect histogram sum, exiting..\n");return(0);
   }

   histogramRRsum = 0L;
   for ( int i = 0; i < binsperdegree*totaldegrees;++i )
        histogramRRsum += histogramRR[i];
   printf("   RR histogram sum = %ld\n",histogramRRsum);
   if ( histogramRRsum != 10000000000L ) {
      printf("   Incorrect histogram sum, exiting..\n");return(0);
   }

   printf("   First Five Omega values:");

   output_file = fopen(argv[3],"w");

   if ( output_file == NULL ) {
      printf("Cannot open output file %s\n",argv[3]);return(-1);
   }
   fprintf(output_file,"bin start\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
   for ( int i = 0; i < binsperdegree*totaldegrees; ++i )
       {
       if ( histogramRR[i] > 0 )
          {
          double omega =  ((double)histogramDD[i]-2* (double)histogramDR[i]+(double)histogramRR[i])/((double)(histogramRR[i]));

          fprintf(output_file,"%6.3f\t%15f\t%15ld\t%15ld\t%15ld\n",((float)i)/binsperdegree, omega,
             histogramDD[i], histogramDR[i], histogramRR[i]);
          if ( i < 5 ) printf("   %6.4lf",omega);
          }
       else
          if ( i < 5 ) printf("         ");
       }

   printf("\n");

   fclose(output_file);

   printf("   Results has been written to the file %s\n",argv[3]);
   printf("   CPU memory allocated  = %.2lf MB\n",CPUMemory/1000000.0);
   printf("   GPU memory allocated  = %.2lf MB\n",GPUMemory/1000000.0);

   gettimeofday(&_ttime, &_tzone);
   walltime = (double)(_ttime.tv_sec) + (double)(_ttime.tv_usec/1000000.0) - walltime;

   printf("   Total wall clock time = %.2lf s\n", walltime);

    
   // free the host and device memory
   free(real_rasc); free(real_decl);
   free(rand_rasc); free(rand_decl);
   cudaFree(d_real_rasc); cudaFree(d_real_decl);
   cudaFree(d_rand_rasc); cudaFree(d_rand_decl);

   free(histogramDR); free(histogramDD); free(histogramRR);
   cudaFree(d_histogramDR); cudaFree(d_histogramDD); cudaFree(d_histogramRR);
    
   return(0);
}

int parseargs_readinput(int argc, char *argv[])
    {
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;
  
    if ( argc != 4 )
       {
       printf("   Usage: galaxy real_data random_data output_file\n   All processes will be killed\n");
       return(1);
       }
    if ( argc == 4 )
       {
       printf("   Running galaxy %s %s %s\n",argv[1], argv[2], argv[3]);

       real_data_file = fopen(argv[1],"r");
       if ( real_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open real data file %s\n",argv[1]);
          return(1);
          }
       else
      {
          fscanf(real_data_file,"%d",&Number_of_Galaxies);
          if ( Number_of_Galaxies != 100000L )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[1]);
             fclose(real_data_file);
             return(1);
             }
          for ( int i = 0; i < NUM_OF_GALAXIES; ++i )
              {
                float rasc, decl;
          if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
             {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[1]);
                 fclose(real_data_file);
             return(1);
             }
          real_rasc[i] = rasc*arcmin2rad;
          real_decl[i] = decl*arcmin2rad;
          }
           fclose(real_data_file);
       printf("   Successfully read 100000 lines from %s\n",argv[1]);
       }

       rand_data_file = fopen(argv[2],"r");
       if ( rand_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open random data file %s\n",argv[2]);
          return(1);
          }
       else
      {
          fscanf(rand_data_file,"%d",&Number_of_Galaxies);
          if ( Number_of_Galaxies != 100000L )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[2]);
             fclose(rand_data_file);
             return(1);
             }
          for ( int i = 0; i < NUM_OF_GALAXIES; ++i )
              {
                float rasc, decl;
          if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
             {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[2]);
                 fclose(rand_data_file);
             return(1);
             }
          rand_rasc[i] = rasc*arcmin2rad;
          rand_decl[i] = decl*arcmin2rad;
          }
          fclose(rand_data_file);
      printf("   Successfully read 100000 lines from %s\n",argv[2]);
      }
       out_file = fopen(argv[3],"w");
       if ( out_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open output file %s\n",argv[3]);
          return(1);
          }
       else fclose(out_file);
       }

    return(0);
    }




int getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio    =    %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                  =    %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim                =    %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                =    %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}
