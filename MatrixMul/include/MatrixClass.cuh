#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

namespace MatrixMul
{
    template<typename T>
    class Matrix{
        private:
            T* _data_A_Host;
            T* _data_B_Host;
            T* _data_C_Host;

            T* _data_A_Device;
            T* _data_B_Device;
            T* _data_C_Device;

            const int _M;
            const int _N;
            const int _K;

            const int _A_size = _M * _N;
            const int _B_size = _N * _K;
            const int _C_size = _M * _N;
        public:
            Matrix(const int M, const int K,const N):_M(M),_K(K),_N(N)
            {
                Initiate_Host(_data_A_Host,M,K);
                Initiate_Host(_data_B_Host,K,N);
                Initiate_Host(_data_C_Host,M,N,0);
                std::cout << "主机端矩阵创建完成，并随机赋值" << std::endl;

                Initiate_Device(_data_A_Device,_A_size,_M,_K);
                Initiate_Device(_data_B_Device,_B_size,_K,_N);
                Initiate_Device(_data_C_Device,_C_size,_M,_N);

                std::cout << "设备端矩阵创建完成" << std::endl;
            }

            void Initiate_Host(T* _data,int row,int col)
            {
                srand(time(NULL));//随机种子
                for(int i = 0; i < row * col; i++)
                {
                    _data[i] = rand() % 100;
                }
                //std::cout << "创建矩阵并初始化成功!!!" << std::endl;
            }

            void Initiate_Host(T* _data, int row,int col,int value)
            {
                for(int i = 0; i < row * col; i++)
                {
                    _data[i] = value;
                }
                //std::cout << "主机端矩阵初始化成功!!!" << std::endl;
            }

            void Initiate_Device(T* _data, int size,int value)
            {
                cudaMalloc((T**)&_data,size * sizeof(T));
            }

            void cudaMem_Host_To_Device()
            {
                cudaMemcpy(_data_A_Device, _data_A_Host, sizeof(T) * _A_size, cudaMemcpyHostToDevice);
                cudaMemcpy(_data_B_Device, _data_B_Host, sizeof(T) * _B_size, cudaMemcpyHostToDevice);
            }
            
            
            ~Matrix()
            {
                delete [] _data_A_Host;
                delete [] _data_B_Host;
                delete [] _data_C_Host;
                std::cout<<"删除主机端矩阵成功"<<std::endl;

            }


            


    };
}