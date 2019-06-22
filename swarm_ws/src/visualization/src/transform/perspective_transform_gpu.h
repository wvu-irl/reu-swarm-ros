#ifndef PERSP_TF_GPU_H
#define PERSP_TF_GPU_H
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

typedef struct
{
	float x;
	float y;
} vector2f_t;

typedef struct
{
	vector2f_t tl; // 0
	vector2f_t tr; // 1
	vector2f_t br; // 2
	vector2f_t bl; // 3
} quadrilateral_t;


__device__ double scale(double val, double o_min, double o_max, double n_min, double n_max);
__device__ vector2f_t warpPoint(quadrilateral_t trap, size_t width, size_t height, vector2f_t initial);

__global__ void transform(sf::Uint8 *cols_in, sf::Uint8 *cols_out, size_t *width, size_t *height, quadrilateral_t *trap);


void createDeviceVar(void **var, size_t size, void *h_var);
void perspectiveTransform(quadrilateral_t trap, sf::RenderTexture *rt, sf::Uint8 *tf_cols);
//#include "perspective_transform_gpu.cu"
#endif
