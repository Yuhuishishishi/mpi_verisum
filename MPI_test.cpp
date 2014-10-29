/*EECS 587 HW3
	by Yuhui Shi
	The entire matrix is transposed */

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cmath>

#define PI 3.141592653589793
#define N_ITER 500

double** arr_mem_loc(const int nrow, const int n);
void arr_mem_clc(double** arr, const int nrow, const int n);
void arr_init_val(double** local_arr, const int rank, const int uni_chunk_size, const int nrow, const int n);
int get_nrow(const int rank, const int numproc, const int n);

int main(int argc, char **argv) {
	// parse the arguments, get the matrix size
	int n;
	if (argc > 1) {
		n = atoi(argv[1]);
	}
	else {
		n = 1000; // defualt value
	}

	int numproc, rank;
	int nrow, uni_chunk_size;
	double time;

	/*******************Initialization*****************/

	// intialize and find the basic info
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	
	// partition the matrix by row, get the partition size
	nrow = get_nrow(rank, numproc, n);
	uni_chunk_size = (int)ceil((double)n / numproc);
	//printf("HERE!!!\n");
	//printf("Chunk size: = %d, nrow = %d\n", uni_chunk_size, nrow);
	//printf("proc:%d, ncol: %d\n", rank, nrow);

	// allocate the memory for the array
	double **local_arr;
	local_arr = arr_mem_loc(nrow, n);
	arr_init_val(local_arr, rank, uni_chunk_size, nrow, n);
	MPI_Barrier(MPI_COMM_WORLD);
	// get the current time
	if (rank == 0)
		time = MPI_Wtime();


	/*******************Iterations****************/
	double* first_row;
	double* last_row;
	double* halo_up;
	double* halo_bt;

	MPI_Request up_request;
	MPI_Request bt_request;
	MPI_Status up_status;
	MPI_Status bt_status;

	// do iterations
	double** new_arr = new double*[nrow];
	for (int i = 0; i < nrow; i++) {
		new_arr[i] = new double[n];
	}

	
	for (int iter = 0; iter < N_ITER; iter++) {
		//printf("Iteration : %d\n", iter);

		first_row = local_arr[1];
		last_row = local_arr[nrow];
		halo_up = local_arr[0];
		halo_bt = local_arr[nrow + 1];
		// send the halo rows to neighbours
		int up_nb_rank = (rank - 1 + numproc) % numproc;
		int bt_nb_rank = (rank + 1 + numproc) % numproc;
		MPI_Isend(first_row, n, MPI_DOUBLE, up_nb_rank, 0, MPI_COMM_WORLD, &up_request);
		MPI_Isend(last_row, n, MPI_DOUBLE, bt_nb_rank, 0, MPI_COMM_WORLD, &bt_request);
		MPI_Recv(halo_up, n, MPI_DOUBLE, up_nb_rank, 0, MPI_COMM_WORLD, &up_status);
		MPI_Recv(halo_bt, n, MPI_DOUBLE, bt_nb_rank, 0, MPI_COMM_WORLD, &bt_status);

		// update the cells
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < n; j++) {
				// do not update the boundary cell
				if (j == 0 || j == n-1)
					new_arr[i][j] = local_arr[i + 1][j];
				else {
					// average of all neighbours and it self
					new_arr[i][j] = (local_arr[i][j - 1] + local_arr[i][j] + local_arr[i][j + 1]
						+ local_arr[i + 1][j - 1] + local_arr[i + 1][j] + local_arr[i + 1][j + 1]
						+ local_arr[i + 2][j - 1] + local_arr[i + 2][j] + local_arr[i + 2][j + 1])/9.0;
				}
			}
		}


		// copy back the values
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < n; j++) {
				local_arr[i + 1][j] = new_arr[i][j];
			}
		}
	}

	// clear the memory
	for (int i = 0; i < nrow; i++) { delete[] new_arr[i]; }
	delete[] new_arr;

	/**************************Calulate the verification sum*****************************/
	// calculate the local verification sum
	int global_i;
	double local_veri_sum = 0;
	for (int i = 1; i < nrow + 1; i++) {
		global_i = uni_chunk_size * rank + i - 1;
		for (int j = 0; j < n; j++) {
			if (global_i == j)
				local_veri_sum += local_arr[i][j];
		}
	}

	// add up all local verification sum
	double total_veri_sum = 0;
	MPI_Reduce(&local_veri_sum, &total_veri_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	// calculate and print out the time
	if (rank == 0) {
		time = MPI_Wtime() - time;
		printf("Verification sum = %.6f, Wall time = %.2f\n", total_veri_sum, time);
	}

	

	arr_mem_clc(local_arr, nrow, n);
	MPI_Finalize();

	return 0;
}

int get_nrow(const int rank, const int numproc, const int n) {
	int nrow;
	nrow = (int)ceil((double)n / numproc);
	if (n % nrow != 0) {
		if (rank == numproc - 1) {
			nrow = n % nrow;
		}
	}
	return nrow;
}

double** arr_mem_loc(const int nrow, const int n) {
	double** arr = new double*[nrow + 2]; // add two halo rows
	for (int i = 0; i < nrow + 2; i++) {
		arr[i] = new double[n]; 
	}
	return arr;
}

void arr_mem_clc(double** arr, const int nrow, const int n) {
	for (int i = 0; i < nrow+2; i++) {
		delete[] arr[i];
	}
	delete[] arr;
}

void arr_init_val(double** local_arr, const int rank, const int uni_chunk_size, const int nrow, const int n) {
	int global_i; 
	for (int i = 1; i < nrow + 1; i++) {
		global_i = uni_chunk_size * rank + i - 1;
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				local_arr[i][j] = 0;
			}
			else if (j == n - 1) {
				local_arr[i][j] = 5 * sin(PI * pow(global_i/(double)n, 2));
				//local_arr[i][j] = 123;
				//printf("%.4f\n", local_arr[i][j]);
			}
			else {
				local_arr[i][j] = 0.5;
			}
		}
	}
}
