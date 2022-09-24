#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "../include/helper_math.h"
#include "../include/helper_geometry_gpu.h"

#define BLOCKSIZE_X           16
#define BLOCKSIZE_Y           16


texture<float, cudaTextureType3D, cudaReadModeElementType> volume_as_texture;
texture<float, cudaTextureType3D, cudaReadModeElementType> sinogram_as_texture;
#define CUDART_INF_F __int_as_float(0x7f800000)

namespace {


__device__ float kernel_project3D(const float3 source_point, const float3 ray_vector,
                                  const uint3 volume_size, const float3 volume_origin, const float volbiasz, const float dSampleInterval, const float dSliceInterval)
{
    float pixel = 0.0f;
    // Step 1: compute alpha value at entry and exit point of the volume
    float min_alpha, max_alpha;
    min_alpha = 0;
    max_alpha = CUDART_INF_F;

    if (fabs(ray_vector.x) >= fabs(ray_vector.y))
    {
        float volume_min_edge_point = volume_origin.x;
        float volume_max_edge_point = volume_size.x + volume_origin.x;

        float reci = 1.0f / ray_vector.x;
        float alpha0 = (volume_min_edge_point - source_point.x) * reci;
        float alpha1 = (volume_max_edge_point - source_point.x) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }
    else
    {
        float volume_min_edge_point = volume_origin.y;
        float volume_max_edge_point = volume_size.y + volume_origin.y;

        float reci = 1.0f / ray_vector.y;
        float alpha0 = (volume_min_edge_point - source_point.y) * reci;
        float alpha1 = (volume_max_edge_point - source_point.y) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }

    min_alpha -= 3; //two end
    max_alpha += 3;

    float px, py, pz;
    float step_size = dSampleInterval;

    while (min_alpha < max_alpha)
    {
        px = source_point.x + min_alpha * ray_vector.x;
        py = source_point.y + min_alpha * ray_vector.y;
        pz = source_point.z + min_alpha * ray_vector.z - volbiasz;
        px /= step_size;
        py /= step_size;
        pz /= dSliceInterval;
        px -= volume_origin.x;
        py -= volume_origin.y;
        pz -= volume_origin.z;
        pixel += tex3D(volume_as_texture, px + 0.5f, py + 0.5f, pz + 0.5f);
        min_alpha += step_size;
    }
    // Scaling by stepsize;
    pixel *= step_size;


    return pixel;
}

__global__ void project_3Dcone_beam_kernel( float *pSinogram,
                                            const float2 *d_rays,
                                            const uint3 volume_size, const float3 volume_origin,
                                            const uint2 detector_size, const float2 detector_origin,
                                            const float sid, const float sdd, const float volbiasz, const float fbiasz, const float dSliceInterval,  const uint projidx)
{
    //return;
    uint2 detector_idx = make_uint2( blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y  );
    uint projection_number = projidx;
    //Prep: Wrap pointer to float2 for better readable code
    //float3 volume_spacing = make_float3(*(volume_spacing_ptr+2), *(volume_spacing_ptr+1), *volume_spacing_ptr);
    if (detector_idx.x >= detector_size.x || detector_idx.y >= detector_size.y)
    {
        return;
    }
    // //Preparations:
    // d_inv_AR_matrices += projection_number * 9;
    // float3 source_point = d_src_points[projection_number];
    //Compute ray direction
    float2 central_ray_vector = d_rays[projection_number];
    float2 u_vec = make_float2(-central_ray_vector.y, central_ray_vector.x);
    float  u = detector_idx.x + detector_origin.x;
    float  angle = u / (sdd - sid);
    float  R2 = sdd - sid;
    float2 spoint = central_ray_vector * (-sid);
    float2 detector_point_world = spoint + central_ray_vector * (sid + R2 * cos(angle)) + u_vec * R2 * sin(angle);
    float2 rvec = detector_point_world - spoint;
    float rz = detector_idx.y + detector_origin.y + fbiasz;

    float3 ray_vector = make_float3(rvec.x,rvec.y,rz); ///rz待定
    ray_vector = normalize(ray_vector);
    float3 source_point = make_float3(spoint.x, spoint.y, 0);

    /*if (detector_idx.x ==369 && detector_idx.y == 0 && projidx == 0)
    {
        printf("source point XYZ: %f %f %f\n",source_point.x,source_point.y,source_point.z);
        printf("volume size XYZ: %d %d %d\n",volume_size.x,volume_size.y,volume_size.z);
        printf("ray_vector XYZ: %f %f %f\n",rvec.x,rvec.y,rz);
        printf("grad XYZ: %f %f %f\n",ray_vector.x,ray_vector.y,ray_vector.z);
        //printf("val: %f\n",tex3D(volume_as_texture, 128 + 0.5f, 128 + 0.5f, 6 + 0.5f));
    }*/


    //float dSampleInterval = sid / sdd;
    float theta_dec = detector_size.x / (2*(sdd - sid));
    float BG = (sdd - sid) * sin(theta_dec);
    float AG = (sdd - sid) * cos(theta_dec) + sid;
    float AB = sqrt(BG * BG + AG * AG);
    float dRadiusTemp = BG/AB*sid*2;
    float dSampleInterval = dRadiusTemp/volume_size.x;

    float pixel = kernel_project3D(
        source_point,
        ray_vector,
        volume_size,
        volume_origin,
        volbiasz,
        dSampleInterval,
        dSliceInterval);

    unsigned sinogram_idx = projidx * detector_size.x * detector_size.y + detector_idx.y * detector_size.x + detector_idx.x;

    /*
     pixel *= sqrt(  (ray_vector.x * volume_spacing.x) * (ray_vector.x * volume_spacing.x) +
                     (ray_vector.y * volume_spacing.y) * (ray_vector.y * volume_spacing.y) +
                     (ray_vector.z * volume_spacing.z) * (ray_vector.z * volume_spacing.z)  );
    */

    pSinogram[sinogram_idx] = pixel;
    return;
}




__device__ float CalcuWeight(float3 pixel, float alpha, float gammamax1, float gammamax2, float Rf,float Rm, float flag)
{
        if (flag > 0)
            printf("para alpha:%f  Rf:%f Rm:%f gammamax1-2:%f %f\n",alpha,Rf,Rm, gammamax1, gammamax2);
        const float PI = 3.14159265359f;
	float w1, w2, w_ps, w_t;

	float x = pixel.x;
	float y = pixel.y;
        float z = pixel.z;
	float r = sqrt(x * x + y * y);
	float phi = atan2(-1 * x, y);
	float c;
	float dr = Rm * Rm / (2 * Rf);  // alpha, Rm, Rf未定
	float theta = alpha - atan2(r * sin(alpha - phi), Rf + r * cos(alpha - phi));
	while (theta < -PI)
	{
		theta += 2 * PI;
	}
	while (theta > PI)
	{
		theta -= 2 * PI;
	}
	float theta1 = phi - PI / 2;
	float theta2 = phi + PI / 2;
	float dtheta, angle1, angle2, angle3, angle4;
	float r0;

	//计算体素加权
	w_t = 0;
	w_ps = 0;
	//z = abs(IniSlice - MidY + k * SliceInter);  //z已经给出

	c = 1 / tan(gammamax1);

        if (flag > 0)
            printf("info z:%f  c:%f r:%f, dr:%f phi:%f \n",z,c,r,dr, phi);

        if (Rf - z * c < 0)
	    w1 = 1;
	else
	{
		r0 = Rf - z * c - dr;
		if (r >= r0 + dr)
		{
			w_t = 1;
			//dtheta = asin((Rf*(Rf - z * c) - r * r) / (r*sqrt(Rf*(2 * z * c - Rf) + r * r)));
			dtheta = asin((Rf - z * c) / r) - atan(sqrt(r*r - (Rf - z * c)*(Rf - z * c)) / (z * c));
                        if (flag > 0)
                            printf("info dtheta:%f \n",dtheta);
			while (dtheta < -PI)
			{
				dtheta += 2 * PI;
			}
			while (dtheta > PI)
			{
				dtheta -= 2 * PI;
			}
			angle1 = theta1 - dtheta;
			angle2 = theta2 + dtheta;
			angle3 = theta1 + dtheta;
			angle4 = theta2 - dtheta;
			if (theta - angle2 > PI / 2) theta -= 2 * PI;
			if (angle1 - theta > PI / 2) theta += 2 * PI;
			if (theta >= angle1 && theta < angle3)
			{
				w_ps = 1 + sin(PI* (theta - angle1 - dtheta) / (2 * dtheta));
                                if (flag > 0)
                                    printf("en 1 \n");
			}
			else if (theta <= angle2 && theta > angle4)
			{
				w_ps = 1 - sin(PI* (theta - angle2 + dtheta) / (2 * dtheta));
                                if (flag > 0)
                                    printf("en 2 \n");
			}
			else if (theta >= angle3 && theta <= angle4)
			{
				w_ps = 2;
                                if (flag > 0)
                                    printf("en 3 \n");
			}
		}
		else if (r >= r0 - dr)
		{
			w_t = 0.5 + 0.5 * sin(PI*(r - r0) / (2 * dr));
			w_ps = 1 - cos(theta - phi - PI);
                        if (flag > 0)
                            printf("info wt:%f  w_ps:%f \n",w_t,w_ps);
		}
		w1 = 1 - w_t + w_t * w_ps;
	}
        if (flag > 0)
            printf("info w1:%f theta:%f angle1-4:%f %f %f %f \n",w1,theta,angle1,angle2,angle3,angle4);

	c = 1 / tan(gammamax2);
        r0 = z * c - Rf - dr;
        w_t = 0;
	w_ps = 0;
	if (Rf - z * c > 0)
		w2 = 1;
	else
	{
		// r0 = Rf - z * c - dr;
		if (r >= r0 + dr)
		{
			w_t = 1;
			//dtheta = asin((Rf*(z * c - Rf) - r * r) / (r*sqrt(Rf*(2 * z * c - Rf) + r * r)));
			dtheta = asin((z * c - Rf) / r) - atan(sqrt(r*r - (Rf - z * c)*(Rf - z * c)) / (z * c));
			while (dtheta < -PI)
			{
				dtheta += 2 * PI;
			}
			while (dtheta > PI)
			{
				dtheta -= 2 * PI;
			}
			angle1 = theta1 - dtheta;
			angle2 = theta2 + dtheta;
			angle3 = theta1 + dtheta;
			angle4 = theta2 - dtheta;
			if (theta - angle2 > PI / 2) theta = theta - 2 * PI;
			if (angle1 - theta > PI / 2) theta = theta + 2 * PI;
			if (theta > angle1 && theta < angle3)
			{
				w_ps = 1 + sin(PI* (theta - angle1 - dtheta) / (2 * dtheta));
			}
			else if (theta < angle2 && theta > angle4)
			{
				w_ps = 1 - sin(PI* (theta - angle2 + dtheta) / (2 * dtheta));
			}
			else if (theta >= angle3 && theta <= angle4)
			{
				w_ps = 2;
			}
		}
		else if (r >= r0 - dr)
		{
			w_t = 0.5 + 0.5 * sin(PI*(r - r0) / (2 * dr));
			w_ps = 1 - cos(theta - phi - PI);
		}
		w2 = 1 - w_t + w_t * w_ps;
	}

    return w1 * (2-w2); //// w1 * (2-w2) / 2;

}


__global__ void backproject_3Dcone_beam_kernel( float *vol,
                                            const float2 *d_rays,
                                            int number_of_projections,
                                            const uint3 volume_size, const float3 volume_origin,
                                            const uint2 detector_size, const float2 detector_origin,
                                            const float sid, const float sdd, const float volbiasz,
                                            const float fbiasz, const float dSliceInterval,const uint projidx)

{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;

   //float3 volume_spacing = make_float3(*(volume_spacing_ptr+2), *(volume_spacing_ptr+1), *volume_spacing_ptr); ///????
   const float pi = 3.14159265359f;

   if( i >= volume_size.x  || j >= volume_size.y )
      return;


    float theta_dec = detector_size.x / (2*(sdd - sid));
    float BG = (sdd - sid) * sin(theta_dec);
    float AG = (sdd - sid) * cos(theta_dec) + sid;
    float AB = sqrt(BG * BG + AG * AG);
    float dRadiusTemp = BG/AB*sid*2;
    float dSampleInterval = dRadiusTemp/volume_size.x;

   //const float2 coordinates = make_float2(i+volume_origin.x,j+volume_origin.y) * (sid / sdd);
   const float2 coordinates = make_float2(i+volume_origin.x,j+volume_origin.y) * dSampleInterval;

   float val = 0.0f;

   float2 central_ray = d_rays[projidx];
   float2 detector_vec = make_float2(-central_ray.y, central_ray.x);
   float2 source_position = central_ray * (-(sid));

   float2 pts = coordinates - source_position;
   float2 line_vec = normalize(pts);
   float2 mid_point = dot(-1*source_position, line_vec) * line_vec + source_position;
   float  R2 = sdd - sid;
   float  rc = sqrt(R2*R2 - (mid_point.x*mid_point.x + mid_point.y*mid_point.y));
   float2 dest = mid_point + rc*line_vec;
   float  angle = atan2(dot(dest, detector_vec), dot(dest, central_ray));

   float s_idx = R2 * angle - detector_origin.x;
   float dz = (dSliceInterval) * length(dest-source_position) / length(pts);
   float coff = sid * sdd / length(pts) / length(pts);


   float z = (volume_origin.z + volbiasz)*dz - fbiasz - detector_origin.y; // 初始值

   float alpha = atan2(central_ray.y, central_ray.x) - pi /2; // + pi /2
   //float vSdd  = dot(central_ray, dest-source_position);
   float gammamax1 = atan(abs(fbiasz + detector_origin.y) / sdd); //sdd
   float gammamax2 = atan(abs(fbiasz + detector_origin.y + detector_size.y) / sdd); //sdd

   /*if (i == 127 && j == 127 && projidx == 1)
   {
       printf("info line_vec:%f  %f \n",line_vec.x,line_vec.y);
       printf("info coordinates:%f  %f \n",coordinates.x,coordinates.y);
       printf("info pts:%f  %f \n",pts.x,pts.y);
       printf("info mid_point:%f  %f \n",mid_point.x,mid_point.y);
       printf("info dest:%f  %f \n",dest.x,dest.y);
       printf("info rc:%f \n",rc);
       printf("info dz:%f s_idx:%f z:%f \n",dz,s_idx,z);
   }*/

   /*if (i == 127 && j == 127)
   {
       printf("info gamma1-2:%f  %f \n",gammamax1,gammamax2);
       printf("info central_ray:%f  %f \n",central_ray.x,central_ray.y);
       float3 pixel = make_float3(150, 0, abs(volume_origin.z+20+volbiasz)*dSliceInterval);
       float weight = CalcuWeight(pixel, alpha, gammamax1, gammamax2, sid, 0.5*volume_size.x*(sid / sdd), 1.0);
       printf("weight:%f \n",weight);
   }*/


   for (int k = 0; k < volume_size.z; ++k)
   {
       /*
       if (coordinates.x * coordinates.x + coordinates.y * coordinates.y > 591/2 * 591/2 * (sid/sdd)*(sid/sdd))
       {
           continue;
       }
       */
       float3 pixel = make_float3(coordinates.x, coordinates.y, abs(volume_origin.z+k+volbiasz)*dSliceInterval);
       // float z_cor = k+volume_origin.z;
       int volume_idx = k * volume_size.x * volume_size.y + j * volume_size.x + i;
       float val = tex3D(sinogram_as_texture, s_idx + 0.5f, z + 0.5f, projidx+0.5f);
       //float weight = CalcuWeight(pixel, alpha, gammamax1, gammamax2, sid, 0.5*volume_size.x*(sid / sdd), 0.0);
       //val *= (weight * coff * 2*pi/number_of_projections); //weight *
       //val *= (2*pi/number_of_projections);
       vol[volume_idx] += val;
       //vol[volume_idx] = weight;
       //vol[volume_idx] += tex3D(sinogram_as_texture, s_idx + 0.5f, z + 0.5f, projidx+0.5f);
       z += dz;
   }

}

} // namespace

std::vector<torch::Tensor> cone_project_cuda_forward(
    torch::Tensor volume,
    torch::Tensor ray_vectors,
    int number_of_projections,
    const int volume_width,
    const int volume_height,
    const int volume_depth,
    const float volume_origin_x,
    const float volume_origin_y,
    const float volume_origin_z,
    const int detector_width,
    const int detector_height,
    const float detector_origin_x,
    const float detector_origin_y,
    const float sid,
    const float sdd,
    const float volbiasz,
    const float fbiasz,
    const float dSliceInterval,
    const int   deviceId)
{
  int batchsize = volume.size(0);
  //printf("batchsize: %d\n",batchsize);
  if (batchsize < 1)
  {
      return {torch::zeros({1,number_of_projections,detector_height,detector_width})};
  }

  auto out = torch::zeros({batchsize,number_of_projections,detector_height,detector_width}).to(volume.device());

  cudaSetDevice(deviceId);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  volume_as_texture.addressMode[0] = cudaAddressModeBorder;
  volume_as_texture.addressMode[1] = cudaAddressModeBorder;
  volume_as_texture.addressMode[2] = cudaAddressModeBorder;
  volume_as_texture.filterMode = cudaFilterModeLinear;
  volume_as_texture.normalized = false;

  uint3 volume_size = make_uint3(volume_width, volume_height, volume_depth);
  float3 volume_origin = make_float3(volume_origin_x,volume_origin_y,volume_origin_z);
  uint2 detector_size = make_uint2(detector_width, detector_height);
  float2 detector_origin = make_float2(detector_origin_x,detector_origin_y);

  int idx = 0;
  float* volume_ptr = volume.data<float>();
  float* out_ptr = out.data<float>();
  float* Nvolume_ptr = NULL;
  float* Nout_ptr = NULL;

  for (idx = 0; idx < batchsize; idx++)
  {
      Nvolume_ptr = volume_ptr + volume_size.x * volume_size.y * volume_size.z * idx;
      Nout_ptr = out_ptr + number_of_projections * detector_size.x * detector_size.y * idx;

      cudaExtent m_extent = make_cudaExtent(volume_size.x, volume_size.y, volume_size.z);
      cudaArray *volume_array;
      cudaMalloc3DArray(&volume_array, &channelDesc, m_extent);
      //cudaMemcpyToArray(volume_array, 0, 0, Nvolume_ptr, volume_size_x * volume_size_y * sizeof(float), cudaMemcpyDeviceToDevice);
      //cudaBindTextureToArray(volume_as_texture, volume_array, channelDesc); //体数据放在纹理内存中
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr = make_cudaPitchedPtr((void*)Nvolume_ptr, volume_size.x*sizeof(float), volume_size.x, volume_size.y);
      copyParams.dstArray = volume_array;
      copyParams.kind = cudaMemcpyDeviceToDevice;

      copyParams.extent = m_extent;
      cudaMemcpy3D(&copyParams);
      cudaBindTextureToArray(volume_as_texture, volume_array, channelDesc);


      const dim3 blocksize = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
      const dim3 gridsize = dim3( detector_size.x / blocksize.x + 1, detector_size.y / blocksize.y + 1 , 1);

      int proji = 0;
      for (proji = 0; proji < number_of_projections; ++proji)
      {
           project_3Dcone_beam_kernel<<<gridsize, blocksize>>>(Nout_ptr, ((float2*) ray_vectors.data<float>()),
                                        volume_size,volume_origin, detector_size, detector_origin,sid,sdd,volbiasz,fbiasz,dSliceInterval, proji);
      }

      cudaUnbindTexture(volume_as_texture);
      cudaFreeArray(volume_array);
  }

  return {out};
}




std::vector<torch::Tensor> cone_project_cuda_backward(
                             torch::Tensor sinogram,
                             torch::Tensor ray_vectors,
                             int number_of_projections,
                             const int volume_width,
                             const int volume_height,
                             const int volume_depth,
                             const float volume_origin_x,
                             const float volume_origin_y,
                             const float volume_origin_z,
                             const int detector_width,
                             const int detector_height,
                             const float detector_origin_x,
                             const float detector_origin_y,
                             const float sid,
                             const float sdd,
                             const float volbiasz,
                             const float fbiasz,
                             const float dSliceInterval,
                             const int   deviceId)
{
  int batchsize = sinogram.size(0);
  //printf("batchsize: %d\n",batchsize);
  if (batchsize < 1)
  {
      return {torch::zeros({1,volume_depth,volume_height,volume_width})};
  }
  auto out = torch::zeros({batchsize,volume_depth,volume_height,volume_width}).to(sinogram.device());

  cudaSetDevice(deviceId);
  int idx = 0;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  sinogram_as_texture.addressMode[0] = cudaAddressModeBorder;
  sinogram_as_texture.addressMode[1] = cudaAddressModeClamp;
  sinogram_as_texture.addressMode[2] = cudaAddressModeBorder;//cudaAddressModeClamp;//cudaAddressModeBorder;
  sinogram_as_texture.filterMode = cudaFilterModeLinear;
  sinogram_as_texture.normalized = false;

  float* sinogram_ptr = sinogram.data<float>();
  float* out_ptr = out.data<float>();
  float* Nsinogram_ptr = NULL;
  float* Nout_ptr = NULL;

  uint3 volume_size = make_uint3(volume_width, volume_height, volume_depth);
  float3 volume_origin = make_float3(volume_origin_x,volume_origin_y,volume_origin_z);
  uint2 detector_size = make_uint2(detector_width, detector_height);
  float2 detector_origin = make_float2(detector_origin_x,detector_origin_y);

  for (idx = 0; idx < batchsize; idx++)
  {
      Nsinogram_ptr = sinogram_ptr + number_of_projections * detector_size.x * detector_size.y * idx;
      Nout_ptr = out_ptr + volume_size.x * volume_size.y * volume_size.z * idx;

      cudaExtent m_extent = make_cudaExtent(detector_size.x, detector_size.y, number_of_projections);
      cudaArray *sinogram_array;
      cudaMalloc3DArray(&sinogram_array, &channelDesc, m_extent);
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr = make_cudaPitchedPtr((void*)Nsinogram_ptr, detector_size.x*sizeof(float), detector_size.x, detector_size.y);
      copyParams.dstArray = sinogram_array;
      copyParams.kind = cudaMemcpyDeviceToDevice;

      copyParams.extent = m_extent;
      cudaMemcpy3D(&copyParams);
      cudaBindTextureToArray(sinogram_as_texture, sinogram_array, channelDesc);

      const dim3 blocksize = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
      const dim3 gridsize = dim3( volume_size.x / blocksize.x + 1, volume_size.y / blocksize.y + 1 , 1);


      int proji = 0;
      for (proji = 0; proji < number_of_projections; ++proji) //number_of_projections
      {
          backproject_3Dcone_beam_kernel<<< gridsize, blocksize >>>( Nout_ptr, ((float2*) ray_vectors.data<float>()), number_of_projections,
                                                         volume_size, volume_origin, detector_size, detector_origin, sid, sdd, volbiasz, fbiasz, dSliceInterval, proji);
      }



      cudaUnbindTexture(sinogram_as_texture);
      cudaFreeArray(sinogram_array);
  }

  return {out};
}