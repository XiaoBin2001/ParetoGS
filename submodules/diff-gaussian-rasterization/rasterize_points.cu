/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool inference_only,
	const bool compute_invdepth,
	const torch::Tensor& pixel_weights,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS_3DGS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = inference_only || !compute_invdepth
	? torch::empty({0}, float_opts)
	: torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  auto byte_opts = means3D.options().dtype(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
  torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
  torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
  torch::Tensor sampleBuffer = torch::empty({0}, byte_opts);
  torch::Tensor accumWeightsBuffer = torch::empty({0}, float_opts);
  torch::Tensor colors = torch::empty({0}, float_opts);
  torch::Tensor cov3D_precomp = torch::empty({0}, float_opts);
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

  const float* pixel_weight_ptr = nullptr;
  float* accum_weight_ptr = nullptr;
  if (pixel_weights.numel() != 0)
  {
	accumWeightsBuffer = torch::full({P}, 0.0, float_opts);
	pixel_weight_ptr = pixel_weights.contiguous().data_ptr<float>();
	accum_weight_ptr = accumWeightsBuffer.contiguous().data_ptr<float>();
  }
  
  int rendered = 0;
  int num_buckets = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		dc.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		inference_only || !compute_invdepth ? nullptr : out_invdepth.contiguous().data<float>(),
		antialiasing,
		inference_only,
		compute_invdepth,
		pixel_weight_ptr,
		accum_weight_ptr,
		radii.contiguous().data<int>(),
		debug);
		
		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
  }
	  return std::make_tuple(rendered, num_buckets, out_color, out_invdepth, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, accumWeightsBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	    const torch::Tensor& projmatrix,
		const float tan_fovx,
		const float tan_fovy,
		const torch::Tensor& out_color,
		const torch::Tensor& out_invdepth,
	    const torch::Tensor& dL_dout_color,
		const torch::Tensor& dc,
		const torch::Tensor& sh,
	const torch::Tensor& dL_dout_invdepth,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const bool antialiasing,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS_3DGS}, means3D.options());
  const bool compute_invdepth = out_invdepth.numel() != 0;
  torch::Tensor dL_dinvdepths = compute_invdepth
	? torch::zeros({P, 1}, means3D.options())
	: torch::empty({0}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); // quats {P, 3, 3}
  torch::Tensor colors = torch::empty({0}, means3D.options().dtype(torch::kFloat32));
  torch::Tensor cov3D_precomp = torch::empty({0}, means3D.options().dtype(torch::kFloat32));
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),	
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
		  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
		  out_color.contiguous().data<float>(),
		  compute_invdepth ? out_invdepth.contiguous().data<float>() : nullptr,
		  dL_dout_color.contiguous().data<float>(),
		  compute_invdepth ? dL_dout_invdepth.contiguous().data<float>() : nullptr,
		  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  compute_invdepth ? dL_dinvdepths.contiguous().data<float>() : nullptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  compute_invdepth,
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

void adamUpdateTorch(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &last_seen_step,
	torch::Tensor &lr_history,
	torch::Tensor &visible,
	const float b1,
	const float b2,
	const float eps,
	const int global_step,
	const uint32_t N,
	const uint32_t M
){
	TORCH_CHECK(param.numel() == param_grad.numel(), "param and param_grad must have the same number of elements");
	TORCH_CHECK(param.numel() == exp_avg.numel(), "param and exp_avg must have the same number of elements");
	TORCH_CHECK(param.numel() == exp_avg_sq.numel(), "param and exp_avg_sq must have the same number of elements");
	TORCH_CHECK(last_seen_step.numel() >= N, "last_seen_step must contain at least N elements");
	TORCH_CHECK(visible.numel() >= N, "visible must contain at least N elements");
	TORCH_CHECK(static_cast<uint64_t>(N) * static_cast<uint64_t>(M) == static_cast<uint64_t>(param.numel()), "N * M must match param.numel()");
	if (N == 0 || M == 0)
	{
		return;
	}
	ADAM::adamUpdateTorch(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		last_seen_step.contiguous().data<int>(),
		lr_history.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		b1,
		b2,
		eps,
		global_step,
		N,
		M);
}
