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

#ifndef PERSP_TF_GPU_H
#define PERSP_TF_GPU_H
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

/**
 * This creates a homography transform for a SFML render texture
 * (goes from rectangle to trapezoid)
 *
 * Uses CUDA to accelerate transform with GPU
 *
 * This transform is being used to convert images to be displayed
 * on the air hockey table correctly (the projector does not project a perfect rectangle)
 *
 * the trapezoid that is being projected can be modified through the
 * calibration program in this package
 *
 */

// Vector to be used in the graphics card
// sf::Vectors cannot be passed
typedef struct
{
	float x;
	float y;
} vector2f_t;

// structure containing all four vertices
// of any quadrilateral
typedef struct
{
	vector2f_t tl; // 0
	vector2f_t tr; // 1
	vector2f_t br; // 2
	vector2f_t bl; // 3
} quadrilateral_t;

// device function to scale a number from one range to that of another range
__device__ double scale(double val, double o_min, double o_max, double n_min, double n_max);
/**
 * performs transform on a single point
 *
 *
 **** Parameters ****
 * trap is the resultant trapezoid
 *
 * width and height are the dimensions of the original rectangle
 *
 * initial is the location of the point to transform in it's original coordinate frame
 *
 **** Returns ****
 * the location to draw the same point in the trapezoid
 */
__device__ vector2f_t warpPoint(quadrilateral_t trap, size_t width, size_t height, vector2f_t initial);

/**
 * performs transform on an entire image
 * this function is the transition function from host to device
 *
 **** Parameters ****
 * cols_in is a pointer to the pixel values in the original image
 *
 * cols_out is an image meant to replace the original frame drawing instead
 * 					a quadrilateral with the image transformed
 *
 * width and height are pointers to the original frame's width and height
 *
 * trap is a pointer to the desired quadrilateral output
 *
 */
__global__ void transform(sf::Uint8 *cols_in, sf::Uint8 *cols_out, size_t *width, size_t *height, quadrilateral_t *trap);

// helper function to copy variables to pointers for the device
void createDeviceVar(void **var, size_t size, void *h_var);

/**
 * Host side of transform copying all relevant information to the
 * GPU and managing events and calling the function to start calculation
 */
void perspectiveTransform(quadrilateral_t trap, sf::RenderTexture *rt, sf::Uint8 *tf_cols);

#endif
