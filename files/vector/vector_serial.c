#include <stdio.h>
#include <stdlib.h>

#define N 1000000

void vector_add(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
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
