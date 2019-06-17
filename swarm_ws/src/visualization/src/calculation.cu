#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <tuple>

#include <cuda_runtime.h>
#include "inc/helper_cuda.h"
#include "inc/helper_functions.h"

#include <SFML/Graphics.hpp>
#include <functional>
#include "color_map.h"

#define DEBUG_GPU_CU 1


namespace hyperthread
{

int devID;
cudaDeviceProp deviceProp;

sf::Uint8 *h_cols;
bool allocated = false;

__device__ void zfunc(double *z, double x, double y, double t)
{
    x -= 640;
	y -= 400;
	x /= 50;
	y /= 50;
	*z = (x * x - y * y) * sin(3.14 / 500 * t);
}

typedef struct
{
    sf::Uint8 r;
    sf::Uint8 g;
    sf::Uint8 b;
} color;

__device__ double scale(double val, double o_min, double o_max, double n_min,double n_max)
{
    return (val - o_min) / (o_max - o_min) * (n_max - n_min) + n_min;
}

__device__ color calculateColor(double val, color colors[], double color_levels[], size_t num_cols)
{
	// low edge case
	if (val <= color_levels[0])
		return colors[0];

	// high edge case
	if (val >= color_levels[num_cols - 1])
		return colors[num_cols - 1];

	// mid case
	for (size_t i = 1; i < num_cols; i++)
	{
		// finding the first color to have a greater value than the input
		if (color_levels[i] > val)
		{
			color col;
			// finding scaled RGB value
			col.r = (int) scale(val, color_levels[i], color_levels[i - 1], colors[i].r,
							colors[i - 1].r);
			col.g = (int) scale(val, color_levels[i], color_levels[i - 1], colors[i].g,
							colors[i - 1].g);
			col.b = (int) scale(val, color_levels[i], color_levels[i - 1], colors[i].b,
							colors[i - 1].b);
			return col;
		}
	}
	return (color){127,127,127};
}

__device__ double max(double a, double b)
{
    return a > b ? a : b;
}

__device__ double min(double a, double b)
{
    return a > b ? b : a;
}

__global__ void gpuThread(double *levels, size_t *num_levels, sf::Uint8 *cols, 
    size_t *width, size_t *height, color *colors, double *color_levels, size_t *num_cols, int *t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_id = idx * 4;
    int i = idx / (*width);
    int j = idx % (*width);

    double zc;
    zfunc(&zc, (double)j, (double)i, (double)(*t));
    color c = calculateColor(zc, colors, color_levels, *num_cols);
    cols[col_id] = c.r;
    cols[col_id + 1] = c.g;
    cols[col_id + 2] = c.b;
    cols[col_id + 3] = 255;

    if (num_levels > 0)
    {
        double zz[4];
        zfunc(zz, (double)j, (double)i - 1, (double)(*t));
        zfunc(zz + 1, (double)j, (double)i + 1, (double)(*t));
        zfunc(zz + 2, (double)j - 1, (double)i, (double)(*t));
        zfunc(zz + 3, (double)j + 1, (double)i, (double)(*t));

        for (size_t k = 0;k < *num_levels;k++)
        {
            bool draw = false;
            for (int l = 0;l < 4 && !draw;l++)
            {
                if (min(zc, zz[l]) <= levels[k] && levels[k] <= max(zc, zz[l]))
                {
                    draw = true;   
                }
            }

            if (draw)
            {
                cols[col_id] = 0;       // R
                cols[col_id + 1] = 0;   // G
                cols[col_id + 2] = 0;   // B
                cols[col_id + 3] = 0;   // A
                break;
            }
        }
    }
}

void createDeviceVar(void **var, size_t size, void *h_var)
{
    std::cout << "\033[31;1m" << std::flush;
    checkCudaErrors(cudaMalloc(var, size));
    checkCudaErrors(cudaMemset(*var, 0, size));
    std::cout << "\033[0m" << std::flush;

    cudaMemcpyAsync(*var, h_var, size, cudaMemcpyHostToDevice, 0);
}

void calc(sf::Uint8 *cols, std::vector<double> levels, size_t width, 
    size_t height, color colors[], double color_levels[], size_t num_cols, int t)
{
    size_t n = width * height;
    size_t size = n * 4;

    // setting up device pointers
    sf::Uint8 *d_cols;
    
    double *d_levels;
    size_t *d_num_levels;
    
    size_t *d_width, *d_height;
    
    color *d_colors;
    double *d_color_levels;
    size_t *d_num_colors;

    int *d_time;

    size_t n_levels = levels.size();

    createDeviceVar((void **)&d_cols, size, cols);
    createDeviceVar((void **)&d_levels, sizeof(double) * levels.size(), levels.data());
    createDeviceVar((void **)&d_num_levels, sizeof(size_t), &n_levels);
    createDeviceVar((void **)&d_width, sizeof(width), &width);
    createDeviceVar((void **)&d_height, sizeof(height), &height);
    createDeviceVar((void **)&d_colors, sizeof(color) * num_cols, colors);
    createDeviceVar((void **)&d_color_levels, sizeof(double) * num_cols, color_levels);
    createDeviceVar((void **)&d_num_colors, sizeof(size_t), &(num_cols));
    createDeviceVar((void **)&d_time, sizeof(int), &t);

    dim3 threads(1024, 1);
    dim3 blocks(n / threads.x, 1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    cudaEventRecord(start, 0);
    gpuThread<<<blocks, threads>>>(
        d_levels,
        d_num_levels,
        d_cols, 
        d_width, 
        d_height, 
        d_colors, 
        d_color_levels, 
        d_num_colors,
        d_time);
    cudaMemcpyAsync(cols, d_cols, size, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        usleep(100);
    }

    std::cout << "\033[31;1m" << std::flush;
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_cols));
    checkCudaErrors(cudaFree(d_levels));
    checkCudaErrors(cudaFree(d_num_levels));
    checkCudaErrors(cudaFree(d_width));
    checkCudaErrors(cudaFree(d_height));
    checkCudaErrors(cudaFree(d_colors));
    checkCudaErrors(cudaFree(d_color_levels));
    checkCudaErrors(cudaFree(d_num_colors));
    checkCudaErrors(cudaFree(d_time));
    std::cout << "\033[0m" << std::flush;
}

void init()
{
    const char *str = "";
    devID = findCudaDevice(0, &str);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
}

void destruct()
{
}

}