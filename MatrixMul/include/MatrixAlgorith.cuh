#include <stdio.h>

namespace MatrixAlgorith {

    using ElemType = float;
    /* 最原始的三层for循环计算AxB=C, M->N->K */
    void MatrixMulOrigin(const ElemType* A, const ElemType *B, ElemType *C, const int& M, const int& K, const int& N)
    {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                for(int k = 0; k < K; k++)
                    C[i*N+j] += A[i*K+k] * B[k*N+j];
            }
        }
    }
 

}