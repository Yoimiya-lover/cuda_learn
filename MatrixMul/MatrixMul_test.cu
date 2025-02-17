/*
*/

#include <iostream>
#include <stdio.h>
#include "MatrixClass.cuh"
#include "MatrixAlgorith.cuh"
#include "tool.cuh"

#define ROLL_NUM 200

int main(int argc,const char* argv[])
{
    if(argc < 4) {
        std::cout << "参数设置错误, 至少需要3个参数!!!" << std::endl;
        return -1;
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    if(M <=0 || K <= 0 || N <= 0 ) {
        std::cout << "参数设置错误！！！" << "正确参数为: M > 0; K > 0; N > 0; Thread_num >= 0 !!!" << std::endl;
    }
    setGPU();
    MatrixMul::Matrix<float> mat(M,K,N);
    

}