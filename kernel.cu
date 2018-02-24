

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "sm_20_atomic_functions.h"

#include <iostream>
#include "windows.h"
#include <math.h>  
#include <fstream>
#include <random>
#include "main.h"
#include <ctime>
#include <sys/timeb.h>

#include <stdio.h>

using namespace std;

/****************************************************/
//					AZZAOUI YOUSSEF 
// AMERICAN CALL OPTIONS PREMIUM PRICE CALCULATOR
//
//				Mini-Projet EPFL 2015/2016
//							youssef.azzaoui@epfl.ch
/****************************************************/

// Computation parameters
bool CUDAMODE = true;// Cuda vs Cpu mode
const int GPUDEVICE = 1; // device ID of the GPU 
int CUDATHREADS = 512; // should be a multiple of 32 (warp size)
int nRandomWalks = 32768; //32768  must be a multiple of CUDATHREADS
								// and must be superior of 10000 for valid results
int degree = 20;// must be superior to 3 for valid results
// Other const parameters
const double rate = 0.06 / 365; 
const double volatility = 0.2 / 365;

/* End Parameters*/


cudaError_t generateCUDA(int nStep, double initialPrice, int stepTime);

ofstream logFile;
const double expFactor = (rate - (volatility*volatility) / 2);

double* underlyingPrice;//[randomW][step]
double* CUDAunderlyingPrice;//[randomW][step]
double* CUDAA = 0;
double* CUDAAtA = 0;

double* premiumPrice;//[step][randomW]
unsigned long cTime = 1;

int programStart = 0;
int parallelTime = 0;
int computePow = 0;
int gpuCommTime = 0;
int gpuKernelTime = 0;
int gpuTime = 0;
int cholInv = 0;
bool matrixAFirstTime = true;
bool shellMode = false;
int getMilliCount(){
	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

int getMilliSpan(int nTimeStart){
	int nSpan = getMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}


matrix matrixMultiply(pmatrix leftMat, pmatrix rightMat)
{
	int startTime = getMilliCount();
	matrix resMatrix = {};
	resMatrix.height = leftMat->height;
	resMatrix.width = rightMat->width;
	initializeMatrix(&resMatrix);

	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;

	while (i < resMatrix.height)
	{
		j = 0;
		while (j < resMatrix.width)
		{
			double sum = 0;
			k = 0;
			while (k < leftMat->width)
			{ 
				sum += leftMat->val[i*leftMat->width + k] * rightMat->val[k * rightMat->width + j];
				k++;
			}
			resMatrix.val[i*resMatrix.width + j] = sum;
			j++;
		}
		i++;
	}
	parallelTime += getMilliSpan(startTime);
	return resMatrix;
}




//American Call option payout function
double payout(double underlying, double strike)
{

	// Rq : pour les options de type PUT (vente) il suffit 
	// simplement de retourner (strike-underlying)+ 
	if (underlying> strike)
		return underlying - strike;
	else
		return 0;
}

/* Underlying price generation
	CPU VERSION
*/

void underlyingGenerate(double initialPrice, int nStep, double stepTime)
{
	int startTime = getMilliCount();

	std::default_random_engine generator(time(0));
	std::normal_distribution<double> distribution(0, stepTime);

	underlyingPrice = new double[nRandomWalks * nStep];
	for (int i = 0; i<nRandomWalks; ++i)
	{
		
		underlyingPrice[i* nStep] = initialPrice;
		for (int j = 1; j<nStep; ++j)
		{
			double randomNormal = distribution(generator);
			underlyingPrice[i*nStep + j] = underlyingPrice[i*nStep + j - 1] * exp(expFactor* stepTime + volatility*randomNormal);
		}
	}

	//Save data 
	//for matlab verification
	// (Old versions matrix / to update)
	/*
	for (int j=0; j<nStep; ++j)
	{
	logFile << j << "\t";
	for(int i=0;i<nRandomWalks;++i)
	{
	logFile << underlyingPrice[i][j] <<"\t";
	}
	logFile << "\n";
	}
	*/
	int milliSecondsElapsed = getMilliSpan(startTime);
	cout << "Generation elapsed time: " << milliSecondsElapsed << endl;
}

/* Underlying price generation
GPU VERSION
*/
__global__ void generateKernel(double *result, double initialPrice, int nStep, double stepTime, int simPerThread, double expFac,double vol)
{


	int id = threadIdx.x;

	// the XORWOW generator seems to have the best speed for double precision numbers
	curandStateXORWOW curState;
	curand_init(133, id, 0, &curState);

	int firstRandomWalk = id * simPerThread;
	for (int i = firstRandomWalk; i < firstRandomWalk+simPerThread; i++)
	{

		result[i* nStep] = initialPrice;
		for (int j = 1; j<nStep; ++j)
		{
			double randomNormal = curand_normal_double(&curState);
			result[i*nStep + j] = result[i*nStep + j - 1] * exp(expFac* stepTime + vol*randomNormal);
		}
	}

}




//Standard forward substitution algorithm for matrix Inversion
matrix inverseLowTri(pmatrix M)
{

	matrix result = {};
	result.height = M->height;
	result.width = M->width;
	int size = M->width;
	initializeMatrix(&result, 0);
	int i, j, k;

	for (j = 0; j<size; j++)
	{
		result.val[j*size+j] = 1. / M->val[j*size + j];

		for (i = 0; i<j; i++)
		{
			for (k = 0; k<j; k++)
			{
				result.val[j*size + i] += result.val[k* size + i] * M->val[j * size + k];
			}
		}

		for (k = 0; k<j; k++)
		{
			result.val[j*size + k] /= -M->val[j*size + j];
		}
	}

	return result;
}
//Solve Mx=y  with L low triangular and directly because less floating errors
matrix solveLowTri(pmatrix M, pmatrix y)
{
	int startTime = getMilliCount();

	matrix result = {};
	result.height = M->height;
	result.width = 1;
	int size = M->height;
	initializeMatrix(&result, 0);
	int i, j;

	for (j = 0; j<size; j++)
	{

		result.val[j] = y->val[j];

		for (i = 0; i<j; i++)
		{
			result.val[j] -= M->val[j*size+ i] * result.val[i];
		}

		result.val[j] /= M->val[j*size + j];
	}

	cholInv += getMilliSpan(startTime);
	return result;
}

matrix solveHighTri(pmatrix M, pmatrix y)
{
	int startTime = getMilliCount();

	matrix result = {};
	result.height = M->height;
	result.width = 1;
	int size = M->height;
	initializeMatrix(&result, 0);
	int i, j;

	for (j = size - 1; j >= 0; j--)
	{

		result.val[j] = y->val[j];

		for (i = size - 1; i>j; i--)
		{
			result.val[j] -= M->val[j*size+i] * result.val[i];
		}

		result.val[j] /= M->val[j*size+j];
	}
	cholInv += getMilliSpan(startTime);
	return result;
}
void initializeMatrix(pmatrix M)
{
	M->val = new double[M->height * M->width];
	/* Old version array 
	for (int i = 0; i < M->height; i++)
	{
		M->val[i] = new double[M->width];
	}
	*/
}
void initializeMatrix(pmatrix M, double defValue)
{
	M->val = new double[M->height * M->width];

	for (int i = 0; i < M->height; i++)
	{
		for (int j = 0; j < M->width; j++)
		{
			M->val[i*M->width + j] = defValue;
		}
	}
}
void printMatrix(matrix M)
{
	cout.precision(17);
	for (int i = 0; i < M.height; i++)
	{
		for (int j = 0; j < M.width; j++)
		{
			cout << fixed << M.val[i * M.width + j] << "\t";
		}
		cout << endl;
	}
}

matrix transpose(pmatrix M)
{
	matrix result = {};
	result.height = M->width;
	result.width = M->height;
	initializeMatrix(&result);

	for (int i = 0; i < result.height; i++)
	{
		for (int j = 0; j < result.width; j++)
		{
			result.val[i*result.width + j] = M->val[j*M->width + i];
		}
	}
	return result;
}

__global__ void generateMatrixAKernel(double *A, double* underlyingPrice,int degree, int nStep, int date, int simPerThread)
{

	int id = threadIdx.x;
	int firstRandomWalk = id * simPerThread;
	for (int i = firstRandomWalk; i < firstRandomWalk + simPerThread; i++)
	{

		for (int k = 0; k < degree; k++)
		{
			A[i*degree + k] = pow(underlyingPrice[i*nStep + (date - 1)], (double)k);
		}
	}

}
__device__ double atomicAdd2(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void computeAtAKernel(double* A, double * Ata, int degree, int nRandomWalks, int simPerThread)
{

	int id = threadIdx.x;
	int firstRandomWalk = id * simPerThread;

		for (int i = 0; i< degree; i++)
		{
			for (int j = 0; j< degree; j++)
			{
				double sum = 0;
				int I = (i + id) % degree;
				int J = (j + id / degree) % degree;
				for (int k = firstRandomWalk; k < firstRandomWalk + simPerThread; k++)
				{
					sum += A[k*degree + I] * A[k *degree + J];
				}
				atomicAdd2(&Ata[I*degree + J], sum); //avoid races conditions 
			}
		
	}

}




matrix condExpectation(int maxDegree, int date, int nStep)
{
	double res = 0;

	matrix cholL = {};
	matrix cholD = {};

	matrix A = {};
	A.width = maxDegree + 1;
	A.height = nRandomWalks;
	initializeMatrix(&A);

	matrix B = {};
	B.width = nRandomWalks;
	B.height = 1;
	//B.val = &(premiumPrice[date * nRandomWalks]);
	initializeMatrix(&B);
	for (int i = 0; i < nRandomWalks; i++)
		B.val[i] = premiumPrice[date* nRandomWalks + i];




	//printMatrix(B);
	matrix ata;
	if (CUDAMODE)
	{
		ata = generateMatrixAandTransposedAthenMultiplyCUDA(&A, maxDegree + 1,nRandomWalks,nStep,date);
	}
	else
	{
		int startPow = getMilliCount();
		for (int i = 0; i < A.height; i++)
		{
			for (int k = 0; k < A.width; k++)
			{
				A.val[i*A.width + k] = (double)pow((double)underlyingPrice[i*nStep + (date - 1)], k);
			}
		}
		computePow += getMilliSpan(startPow);
		parallelTime += getMilliSpan(startPow);
	    ata = matrixMultiply(&transpose(&A), &A);
	}


	//big matrix to compute

	//matrix cho_ata = cholesky(&ata);// sa roule

	cholesky2(&ata, &cholL, &cholD);//opérationnel enfin

	matrix atb = matrixMultiply(&B, &A);
	matrix sol = transpose(&atb);

	matrix inv_prem = solveLowTri(&cholL, &sol);

	//DL*
	matrix dl = matrixMultiply(&cholD, &transpose(&cholL));
	matrix fsol = solveHighTri(&dl, &inv_prem);


	matrix final = matrixMultiply(&A, &fsol);

	// Debug helpers

	/*cout << "**************************" << endl;
	printMatrix(final);
	//exit(1);*/
	//test:
	/*
	cout << "**************************" << endl;
	cout << "A" << endl;
	printMatrix(A);
	cout << "**************************" << endl;
	cout << "ATA" << endl;
	printMatrix(ata);


	cout << "**************************" << endl;
	cout << "Chol L" << endl;
	printMatrix(cholL);

	cout << "**************************" << endl;
	cout << "Chol D" << endl;
	printMatrix(cholD);

	cout << "**************************" << endl;
	cout << "B*" << endl;
	printMatrix(B);

	cout << "**************************" << endl;
	cout << "inv_prem" << endl;
	printMatrix(inv_prem);

	cout << "**************************" << endl;
	cout << "sol" << endl;
	printMatrix(fsol);

	cout << "**************************" << endl;
	cout << "final" << endl;
	printMatrix(final);
	
	//exit(1);
	*/
	// A faire : changer struc en class + creer destructeur pour faciliter
	free(A.val);
	free(B.val);
	return final;
}

/* 
//No more used because of rounding errors 
Compute L with M = LL*
matrix cholesky(pmatrix M)
{
	int startTime = getMilliCount();

	matrix result = {};
	result.height = M->height;
	result.width = M->width;
	int n = M->width;
	initializeMatrix(&result);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < (i + 1); j++)
		{
			double s = 0.;
			for (int k = 0; k < j; k++)
			{
				s += result.val[i][k] * result.val[j][k];
			}
			if (i == j)
			{

				result.val[i][j] = (M->val[i][i] - s >= 0.) ? sqrt(M->val[i][i] - s) : 0;
			}
			else
			{

				result.val[i][j] = (M->val[i][j] - s) / result.val[j][j];
			}
		}
		for (int j = i + 1; j<n; j++)
			result.val[i][j] = 0.;
	}

	cholInv += getMilliSpan(startTime);
	return result;
}
*/

// Compute L and D with M = LDL* 
// L is low triangular and D is diagonal
void cholesky2(pmatrix M, pmatrix L, pmatrix D)
{
	int startTime = getMilliCount();
	int n = M->width;

	L->height = M->height;
	L->width = M->width;
	initializeMatrix(L, 0.);

	D->height = M->height;
	D->width = M->width;
	initializeMatrix(D, 0.);


	for (int i = 0; i < n; i++)
	{
		double d = M->val[i*M->width + i];
		L->val[i*L->width + i] = 1.;
		for (int j = 0; j < i; j++)
		{
			d -= L->val[i*L->width + j] * L->val[i*L->width + j] * D->val[j*D->width + j];
		}
		D->val[i*D->width + i] = d;
		for (int j = i + 1; j < n; j++)
		{
			double t = M->val[j*M->width + i];
			for (int k = 0; k < i; k++)
			{
				t -= L->val[i*L->width + k] * L->val[j*L->width + k] * D->val[k*D->width + k];
			}
			L->val[j*L->width + i] = t / D->val[i*D->width + i];
		}

	}
	cholInv += getMilliSpan(startTime);
}



// Core of the algorithm
double LSA(double initialPrice, int nStep, int stepTime, double strike)
{
	double res = 0;
	
	if (CUDAMODE)
		generateCUDA(nStep, initialPrice,stepTime);
	else
		underlyingGenerate(initialPrice, nStep, stepTime);

	//initialize price mat
	//premiumPrice[step][randomW]
	matrix premium = {};
	premium.height = nStep;
	premium.width = nRandomWalks;
	initializeMatrix(&premium);
	premiumPrice = premium.val;
	for (int i = 0; i < nRandomWalks; ++i)
		premiumPrice[(nStep - 1)*nRandomWalks + i] = payout(underlyingPrice[i*nStep+(nStep - 1)], strike);

	//printMatrix(premium);
	for (int j = nStep - 1; j>1; j--)
	{
		matrix condExp = condExpectation(degree, j,nStep);
		for (int i = 0; i<nRandomWalks; i++)
		{
			double arb1 = exp(-rate*stepTime) * condExp.val[i*condExp.width];
			double arb2 = payout(underlyingPrice[i* nStep + (j - 1)], strike);
			premiumPrice[(j - 1)*nRandomWalks + i] = arb1 >= arb2 ? arb1 : arb2;
		}
	}
	//last iter
	double cond = 0.;
	for (int i = 0; i<nRandomWalks; i++)
	{
		cond += premiumPrice[1*nRandomWalks + i];
	}
	cond /= nRandomWalks;
	double arb1 = exp(-rate*stepTime) * cond;
	double arb2 = payout(underlyingPrice[0], strike);

	res = arb1 >= arb2 ? arb1 : arb2;

	return res;
}

/*
// Only to verify if the algorithms are working properly
void testCholesky()
{
	matrix test = {};
	test.width = 3;
	test.height = 3;
	initializeMatrix(&test);
	test.val[0][0] = 40.;
	test.val[0][1] = 12.;
	test.val[0][2] = -16.;
	test.val[1][0] = 12.;
	test.val[1][1] = 37.;
	test.val[1][2] = -43.;
	test.val[2][0] = -16.;
	test.val[2][1] = -43.;
	test.val[2][2] = 98.;


	matrix test2 = {};
	test2.width = 1;
	test2.height = 3;

	matrix cholL = {};
	matrix cholD = {};

	initializeMatrix(&test2);
	test2.val[0][0] = 1;
	test2.val[1][0] = 2;
	test2.val[2][0] = 3;

	cout << "Test matrix" << endl;
	printMatrix(test);
	cout << "*****************************" << endl;

	cout << "Choleski" << endl;
	matrix ch = cholesky(&test);
	printMatrix(ch);
	cout << "*****************************" << endl;

	cout << "testBack" << endl;
	matrix testb = solveLowTri(&ch, &test2);
	printMatrix(testb);
	cout << "*****************************" << endl;

	cout << "testBack2" << endl;
	matrix testb2 = solveHighTri(&transpose(&ch), &test2);
	printMatrix(testb2);
	cout << "*****************************" << endl;

	cout << "Inverse" << endl;
	matrix inv = inverseLowTri(&ch);
	printMatrix(inv);
	cout << "*****************************" << endl;
	cout << "test inverse" << endl;
	matrix resultInv = matrixMultiply(&inv, &ch);
	printMatrix(resultInv);
	cout << "*****************************" << endl;
	cout << "Choleski" << endl;
	matrix result = matrixMultiply(&ch, &transpose(&ch));

	printMatrix(result);
}
*/





// Helper function for CUDA
/*Call the underlying generate kernel

*/
cudaError_t generateCUDA(int nStep,double initialPrice,int stepTime)
{
	int startTime = getMilliCount();


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(GPUDEVICE);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&CUDAunderlyingPrice, nRandomWalks * nStep * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		goto Error;
	}

	// on call ici notre premier kernel
	// avec uniquement un bloc pour simplifier les choses
	// avec 1000 threads l'occupancy est de l'ordre de 50% avec une 965M ( 2000 Threads concurrents max)
	// (conclusion faite après profiling)
	// ce qui est correct pour une optique d'analyse du speedup 

	generateKernel <<<1, CUDATHREADS >>>(CUDAunderlyingPrice, initialPrice, nStep, stepTime, nRandomWalks / CUDATHREADS, expFactor, volatility);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "generateKernel launch failed: " <<cudaGetErrorString(cudaStatus) << endl;
		goto Error;
	}

	// cudaDeviceSynchronize barriere de synchro
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching generateKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	underlyingPrice = new double[nRandomWalks * nStep];
	cudaStatus = cudaMemcpy(underlyingPrice, CUDAunderlyingPrice, nRandomWalks * nStep * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int milliSecondsElapsed = getMilliSpan(startTime);
	cout << "GPU Generation elapsed time: " << milliSecondsElapsed << endl;

	Error:
	return cudaStatus;
}

matrix generateMatrixAandTransposedAthenMultiplyCUDA(pmatrix A, int degree, int nRandomWalks, int  nStep, int date)
{
	int startTime2 = getMilliCount();
	cudaError_t cudaStatus;

	matrix ata = {};
	
	if (matrixAFirstTime)
	{
		cudaStatus = cudaMalloc((void**)&CUDAA, nRandomWalks * degree * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMalloc A failed!" << endl;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&CUDAAtA, degree * degree * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMalloc AtransposedA failed!" << endl;
			goto Error;
		}
	}
	int startTime3 = getMilliCount();
	generateMatrixAKernel<<<1, CUDATHREADS >>>(CUDAA, CUDAunderlyingPrice, degree,nStep, date, nRandomWalks / CUDATHREADS);
	
	if (matrixAFirstTime)
	{
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cout << "generateMatrixAKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
			goto Error;
		}
	}
	// cudaDeviceSynchronize barriere de synchro
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching generateMatrixAKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.

	int startTime = getMilliCount();

	cudaStatus = cudaMemcpy(A->val, CUDAA, nRandomWalks * degree  * sizeof(double) , cudaMemcpyDeviceToHost);
	gpuCommTime += getMilliSpan(startTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	computeAtAKernel << <1, CUDATHREADS >> >(CUDAA, CUDAAtA, degree, nRandomWalks, nRandomWalks / CUDATHREADS);
	
	ata.height = degree;
	ata.width = degree;
	ata.val = new double[degree*degree];

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeAkernel\n", cudaStatus);
		goto Error;
	}
	gpuKernelTime += getMilliSpan(startTime3);
	cudaStatus = cudaMemcpy(ata.val, CUDAAtA, degree * degree  * sizeof(double) , cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//reset memory
	cudaMemset(CUDAA, 0, nRandomWalks * degree  * sizeof(double));
	cudaMemset(CUDAAtA, 0, degree * degree  * sizeof(double));

	matrixAFirstTime = false;
	gpuTime += getMilliSpan(startTime2);
	return ata;

Error:
	cudaFree(CUDAA);
	cudaFree(CUDAAtA);
	return ata;
	
}

int main(int argc, char** argv)
{
	cout << "*****************************************************************" << endl;
	cout << "Azzaoui Youssef Monte Carlo Simulator" << endl;
	cout << "Usage : PremiumCalculator.exe nRandomWalks degree [CudaThreads] " << endl ;
	cout << "*****************************************************************" << endl;

	if (argc >= 3)
	{
		nRandomWalks = atoi(argv[1]);
		degree = atoi(argv[2]);
		shellMode = true;
		cout << "Shell mode On" << endl;
	}
	else
		cout << "Shell mode Off" << endl;

	if (argc >= 4)
	{
		CUDATHREADS = atoi(argv[3]);
		cout << " <<<<<CUDA mode On>>>>> - Threads: " << CUDATHREADS << endl;
		CUDAMODE = true;

	}
	double initialPrice = 20.;
	double strike = 20.;
	int time = 365;
	int stepTime = 1;

	int nStep = time / stepTime + 1;
	int Gpumem = (2* 8 * nStep * nRandomWalks )/1000000;

	cout << "Initial Price: " << initialPrice << endl;
	cout << "Strike: " << strike << endl;
	cout << "Time before maturity: " << time << " days" << endl;
	cout << "Numbers of simulations: " << nRandomWalks << endl;
	cout << "nStep: " << nStep << endl;
	if (CUDAMODE)
		cout << "GPU Global memory used(MB): " << Gpumem << endl;
	//Log file Config
	logFile.open("data.txt", ios::out | ios::trunc);
	
	// Start the algorithm
	programStart = getMilliCount();
	double res = LSA(initialPrice, nStep, stepTime, strike);
	int milliSecondsElapsed = getMilliSpan(programStart);

	cout << "Results: " << res << endl;
	if (!CUDAMODE)
	{
		cout << "Parallel time: " << parallelTime << endl;
		cout << "Compute power time: " << computePow << endl;
		cout << "Cholesky solving  time: " << cholInv << endl;
		cout << "P factor: " << (double)parallelTime*100. / (double)milliSecondsElapsed << "%" << endl;
	}
	else
	{
		cout << "Data Transfer  time: " << gpuCommTime << endl;
		cout << "GPU kernel time: " << gpuKernelTime << endl;
		cout << "Total GPU  time: " << gpuTime << endl;
	}
	cout << "Total time: " << milliSecondsElapsed << endl;

	logFile.close();


	// cudaDeviceReset must be called before exiting 
	if (CUDAMODE && cudaDeviceReset() != cudaSuccess)
		cout << "cudaDeviceReset failed!" << endl;
	
	if (!shellMode)
		system("PAUSE");

	return true;
}
