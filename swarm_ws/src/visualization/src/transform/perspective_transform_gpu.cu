/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <visualization/perspective_transform_gpu.h>
#include "../inc/helper_cuda.h"
#include "../inc/helper_functions.h"

#include <unistd.h>

__device__ double scale(double val, double o_min, double o_max, double n_min,
		double n_max)
{
	if (o_max == o_min) // special case
		return n_min;
	return ((val - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min; // shifting numeric domains
}

__device__ vector2f_t warpPoint(quadrilateral_t trap, size_t width,
		size_t height, vector2f_t initial)
{
	// finding connected edges
	vector2f_t top = {(float)scale(initial.x, 0.0, width, trap.tl.x, trap.tr.x),
			(float)scale(initial.x, 0.0, width, trap.tl.y, trap.tr.y)};
	vector2f_t bottom = {(float)scale(initial.x, 0.0, width, trap.bl.x, trap.br.x),
			(float)scale(initial.x, 0.0, width, trap.bl.y, trap.br.y)};
	vector2f_t left = {(float)scale(initial.y, 0.0, height, trap.bl.x, trap.tl.x),
			(float)scale(initial.y, 0.0, height, trap.bl.y, trap.tl.y)};
	vector2f_t right = {(float)scale(initial.y, 0.0, height, trap.br.x, trap.tr.x),
			(float)scale(initial.y, 0.0, height, trap.br.y, trap.tr.y)};

	// linear intersection
	double m0 = (right.y - left.y) / (right.x - left.x);
	double m1 = (bottom.y - top.y) / (bottom.x - top.x);
	double unified_x =
			top.x != bottom.x && m0 != m1 && left.x != right.x ?
					(top.y - right.y + right.x * m0 - top.x * m1) / (m0 - m1) : top.x;
	double unified_y =
			left.y != right.y ? (m0 * (unified_x - right.x) + right.y) : left.y;

	return (vector2f_t){(float) unified_x, (float) unified_y};
}

__global__ void transform(sf::Uint8 *cols_in, sf::Uint8 *cols_out, size_t *width, size_t *height, quadrilateral_t *trap)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // finding iteration level
	int row = *height - i / *width;
	int col = i % *width;
	vector2f_t square_pos = {(float) col, (float) row}; // creating vector of the current position in the rectangle
	vector2f_t trap_point = warpPoint(*trap, *width, *height, square_pos); // finding corresponding position
	if ((int) trap_point.x < *width && (int) trap_point.y < *height
			&& (int) trap_point.x >= 0 && (int) trap_point.y >= 0) // safety catch for drawing pixels
	{
		// copying pixel from one frame to another
		int col_ind = i * 4;
		int o_col_ind = ((int)trap_point.y * (*width) + (int)trap_point.x) * 4;

		for (size_t j = 0;j < 4;j++) // getting values for RGBA
		{
			cols_out[o_col_ind + j] = cols_in[col_ind + j];
		}
	}
}

void createDeviceVar(void **var, size_t size, void *h_var)
{
    std::cout << "\033[31;1m" << std::flush;
    checkCudaErrors(cudaMalloc(var, size)); // allocating device memory
    checkCudaErrors(cudaMemset(*var, 0, size)); // writing 0s to location
    std::cout << "\033[0m" << std::flush;

    cudaMemcpyAsync(*var, h_var, size, cudaMemcpyHostToDevice, 0); // copying from host
}

void perspectiveTransform(quadrilateral_t trap, sf::RenderTexture *rt, sf::Uint8 *tf_cols)
{
	// setting up variables that can be passed
	sf::Image img = rt->getTexture().copyToImage(); // getting image
	sf::Uint8 *col_ptr = (sf::Uint8 *)img.getPixelsPtr(); // getting pixel array
	size_t width = img.getSize().x; // getting dimensions
	size_t height = img.getSize().y;

	// copying to calculation
	// device variables
	sf::Uint8 *d_col_in;
	sf::Uint8 *d_col_out;
	size_t *d_width;
	size_t *d_height;
	quadrilateral_t *d_trap;

	// copying data over
	createDeviceVar((void **)&d_col_in, 4 * sizeof(sf::Uint8) * width * height, col_ptr);

	// out does not have an initial host variable and was merely set up
	checkCudaErrors(cudaMalloc((void **)&d_col_out, 4 * sizeof(sf::Uint8) * width * height));
	checkCudaErrors(cudaMemset(d_col_out, 0, 4 * sizeof(sf::Uint8) * width * height));

	createDeviceVar((void **)&d_width, sizeof(size_t), &width);
	createDeviceVar((void **)&d_height, sizeof(size_t), &height);
	createDeviceVar((void **)&d_trap, sizeof(quadrilateral_t), &trap);

	// setting up processing specifics
	dim3 threads(512, 1); // number of threads per block
	dim3 blocks(width * height / threads.x, 1); // number of blocks in data set to be calculated

	// setup start stop events
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaEventRecord(start, 0); // start event trigger
	transform<<<blocks, threads>>>(d_col_in, d_col_out, d_width, d_height, d_trap); // perform calculation
	cudaMemcpyAsync(tf_cols, d_col_out, 4 * sizeof(sf::Uint8) * width * height, cudaMemcpyDeviceToHost, 0); // copy useful data back

	cudaEventRecord(stop, 0); // stop event trigger

	while (cudaEventQuery(stop) == cudaErrorNotReady) // wait for stop
	{
			usleep(100);
	}

	// cleaning up
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFree(d_col_in));
	checkCudaErrors(cudaFree(d_col_out));
	checkCudaErrors(cudaFree(d_width));
	checkCudaErrors(cudaFree(d_height));
	checkCudaErrors(cudaFree(d_trap));
}
