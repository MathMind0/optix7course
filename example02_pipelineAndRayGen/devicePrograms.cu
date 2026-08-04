// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  { /*! for this simple example, this will remain empty */ }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  { /*! for this simple example, this will remain empty */ }



  //------------------------------------------------------------------------------
  // blue-noise dithering via the R2 low-discrepancy sequence (Roberts 2018).
  //
  // The R2 sequence projects successive integer indices into [0,1)^2 using
  // two carefully chosen irrational multipliers. The resulting 2D point set
  // has the same blue-noise-like Fourier characteristics (energy pushed into
  // high frequencies) as a classic void-and-cluster blue-noise texture, but
  // is generated entirely procedurally - no precomputed texture asset needed.
  //------------------------------------------------------------------------------
  static __forceinline__ __device__ float fracf(float x)
  { return x - floorf(x); }

  static __forceinline__ __device__ void blueNoise(uint32_t i, float &r, float &g, float &b)
  {
    // R2 sequence constants: 1/(1+sqrt(2)) and 1/(2+sqrt(3)) style irrationals
    const float a1 = 0.7548776662466927f; // 1 / (1 + sqrt(2))
    const float a2 = 0.5698402909980532f; // 1 / (2 + sqrt(3))

    // two decorrelated blue-noise values in [0,1)
    const float x = fracf((float)i * a1);
    const float y = fracf((float)i * a2);

    r = x;
    g = y;
    // a third, decorrelated channel derived from the first two
    b = fracf(x + y);
  }


  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    if (optixLaunchParams.frameID == 0 &&
        optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) {
      // we could of course also have used optixGetLaunchDims to query
      // the launch size, but accessing the optixLaunchParams here
      // makes sure they're not getting optimized away (because
      // otherwise they'd not get used)
      printf("############################################\n");
      printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
             optixLaunchParams.fbSize.x,
             optixLaunchParams.fbSize.y);
      printf("############################################\n");
    }

    // ------------------------------------------------------------------
    // for this example, produce a simple test pattern:
    // ------------------------------------------------------------------


    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

#if 0
    const int r = (ix % 256);
    const int g = (iy % 256);
    const int b = ((ix+iy) % 256);
#else
    // r/g/b are now blue-noise distributed values in [0,1)
    float fr, fg, fb;
    blueNoise(ix * 17 ^ iy * 53, fr, fg, fb);
    const int r = (int)(fr*255.99f);
    const int g = (int)(fg*255.99f);
    const int b = (int)(fb*255.99f);
#endif

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.fbSize.x;
    optixLaunchParams.colorBuffer[fbIndex] = rgba;
  }
  
} // ::osc
