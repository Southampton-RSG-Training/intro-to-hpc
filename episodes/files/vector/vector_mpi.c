#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

void vector_add(int *a, int *b, int *c, int n)
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = n / size;
    int *a_local = malloc(n_local * sizeof(int));
    int *b_local = malloc(n_local * sizeof(int));
    int *c_local = malloc(n_local * sizeof(int));

    MPI_Scatter(a, n_local, MPI_INT, a_local, n_local, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, n_local, MPI_INT, b_local, n_local, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n_local; i++)
    {
        c_local[i] = a_local[i] + b_local[i];
    }

    MPI_Gather(c_local, n_local, MPI_INT, c, n_local, MPI_INT, 0, MPI_COMM_WORLD);

    free(a_local);
    free(b_local);
    free(c_local);

    MPI_Finalize();
}

int main(void)
{
    int *a = malloc(N * sizeof(int));
    int *b = malloc(N * sizeof(int));
    int *c = malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    vector_add(a, b, c, N);

    printf("Verification (first 5 elements):\n");
    for (int i = 0; i < 5; i++)
    {
        int expected = 3 * i;
        printf("c[%d] = %3d (expected: %3d)\n", i, c[i], expected);
    }

    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
