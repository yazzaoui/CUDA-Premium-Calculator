typedef struct matrix
{
	double* val;
	int height;
	int width;
}matrix, *pmatrix;

double payout(double underlying, double strike);
void underlyingGenerate(double initialPrice, int nStep, double stepTime);
matrix matrixMultiply(pmatrix leftMat, pmatrix rightMat);
void initializeMatrix(pmatrix M);
matrix cholesky(pmatrix M);
matrix inverseLowTri(pmatrix M);
void initializeMatrix(pmatrix M, double defValue);
matrix condExpectation(int maxDegree, int date, int nStep);
matrix solveLowTri(pmatrix M, pmatrix y);
void cholesky2(pmatrix M, pmatrix L, pmatrix D);
matrix generateMatrixAandTransposedAthenMultiplyCUDA(pmatrix A, int degree, int nRandomWalks, int  nStep, int date);