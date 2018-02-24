### CUDA-Premium-Calculator
### Implementation on GPU with CUDA the Longstaff-Schwartz algorithm to compute American stock option prices.

Read theory.pdf for the theoretical analysis

This program has been developped using Visual Studio 2013 with nVIDIA CUDA 7.5
It has been succesfully working with an nVIDIA GTX 965M SLI configuration.
The program should be compiled with the x64 release version flag.

Usage :
This program is confugured to run on the second graphic card on a MultiGPU system.
To change this behavior go line 31 and change the GPUDEVICE value 1 by the device id you intent to use.

To run the program go to the build directory with a console and type:
> PremiumCalculator.exe nRandomWalks degree [CudaThreads]

where nRandomWalks is the number of simulations, should be  > 20000 for valid output
and where degree should be >= 3
CudaThreads is an optional parameter, if provided the program will be launched in CUDA MODE
with "CudaThreads" the number of threads. 
Please make sure that "CudaThreads" divide "nRandomWalks".
CudaThreads should be a multiple of 32 to maximize occupancy.


With the default input parameters a good result should be around 1.16.
To change those parameters you must change them from the source code (inside main() )

If any question please contact me : y@azzaoui.fr

### Created by Azzaoui Youssef 
