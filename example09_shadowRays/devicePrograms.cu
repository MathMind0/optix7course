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
#include <cuda_runtime.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // GGX PBR helper functions
  //------------------------------------------------------------------------------

  // GGX / Trowbridge-Reitz
  // [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
  static __forceinline__ __device__
  float D_GGX(float NdotH, float a)
  {
    float a2 = a*a;
    float d  = 1.f - NdotH*NdotH * (1.f - a2);
    return a2 / (M_PI * d*d);
  }

  // Appoximation of joint Smith term for GGX
  // [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
  static __forceinline__ __device__
  float Vis_SmithJointApprox( float a, float NoV, float NoL )
  {
    float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
    float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
    return 0.5f * rcp( Vis_SmithV + Vis_SmithL );
  }

  // Schlick Fresnel approximation
  static __forceinline__ __device__
  vec3f F_Schlick(float VdotH, vec3f F0)
  {
    float Fc = powf(1.f - VdotH, 5.f);
    return Fc + (1.f - Fc) * F0;
  }

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for shadow-type rays
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__shadow()
  {
    /* not going to be used - we terminate on first hit */
  }
  
  extern "C" __global__ void __anyhit__shadow()
  {
    // Unnecessary code - we terminate on first hit, see the trace flag of shadow ray.
    //vec3f &prd = *(vec3f*)getPRD<vec3f>();
    //prd = vec3f(0.f);
    //optixTerminateRay();
  }

  //------------------------------------------------------------------------------
  // closest hit for radiance rays - PBR shading with shadow
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    
    // ------------------------------------------------------------------
    // gather basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // interpolate texture coordinates
    vec2f tc;
    if (sbtData.texcoord) {
      tc = (1.f-u-v) * sbtData.texcoord[index.x]
         +         u * sbtData.texcoord[index.y]
         +         v * sbtData.texcoord[index.z];
    }

    // ------------------------------------------------------------------
    // compute shading normal N
    // ------------------------------------------------------------------
    vec3f N;
    if (sbtData.normal) {
      N = (1.f-u-v) * sbtData.normal[index.x]
        +         u * sbtData.normal[index.y]
        +         v * sbtData.normal[index.z];
    } else {
      const vec3f &A = sbtData.vertex[index.x];
      const vec3f &B = sbtData.vertex[index.y];
      const vec3f &C = sbtData.vertex[index.z];
      N = normalize(cross(B-A, C-A));
    }
    N = normalize(N);

    // ------------------------------------------------------------------
    // sample PBR material parameters from textures
    // ------------------------------------------------------------------
    
    // albedo
    vec3f albedo = sbtData.color;
    if (sbtData.texture && sbtData.texcoord) {
      vec4f tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
      albedo *= (vec3f)tex;
    }

    // roughness & metallic from combined metallicRoughness texture
    // glTF convention: G channel = roughness, B channel = metallic
    float roughness = 0.5f;
    float metallic = 0.f;
    if (sbtData.metallicRoughnessMap && sbtData.texcoord) {
      vec4f mrTex = tex2D<float4>(sbtData.metallicRoughnessMap, tc.x, tc.y);
      roughness = mrTex.y;   // G channel = roughness
      metallic  = mrTex.z;   // B channel = metallic
      roughness = clamp(roughness, 0.02f, 1.f);
      metallic  = clamp(metallic,  0.f, 1.f);
    }

    // ambient occlusion (default 1.0) - no separate AO texture in PBR Sponza
    float ao = 0.5f;

    // ------------------------------------------------------------------
    // normal map perturbation (tangent space)
    // ------------------------------------------------------------------
    vec3f shadingN = N;
    if (sbtData.normalMap && sbtData.texcoord) {
      const vec3f &A = sbtData.vertex[index.x];
      const vec3f &B = sbtData.vertex[index.y];
      const vec3f &C = sbtData.vertex[index.z];

      const vec3f e1 = B - A;
      const vec3f e2 = C - A;

      const vec2f duv1 = sbtData.texcoord[index.y] - sbtData.texcoord[index.x];
      const vec2f duv2 = sbtData.texcoord[index.z] - sbtData.texcoord[index.x];

      const float invDet = 1.f / (duv1.x * duv2.y - duv1.y * duv2.x);
      
      vec3f T = (e1 * duv2.y - e2 * duv1.y) * invDet;
      T = normalize(T - N * dot(N, T));
      const vec3f Bt = cross(N, T);

      // sample normal map, decode [0,1] -> [-1,1]
      vec4f nmSample = tex2D<float4>(sbtData.normalMap, tc.x, tc.y);
      vec3f tn = vec3f(2.f * nmSample.x - 1.f,
                        2.f * nmSample.y - 1.f,
                        2.f * nmSample.z - 1.f);
      tn = normalize(tn);

      // transform tangent-space normal to object space
      shadingN = normalize(T * tn.x + Bt * tn.y + N * tn.z);
    }

    // ------------------------------------------------------------------
    // transform shading normal from object space to world space
    // (vertex data, TBN frame, and normal map result are all in object space,
    //  but light dir, view dir, etc. are in world space)
    // ------------------------------------------------------------------
    shadingN = normalize(vec3f(
      optixTransformNormalFromObjectToWorldSpace(shadingN)));

    // also get geometry normal in world space for shadow ray offset
    vec3f worldNg = normalize(vec3f(
      optixTransformNormalFromObjectToWorldSpace(N)));

    // ------------------------------------------------------------------
    // compute surface hit point in world space
    // ------------------------------------------------------------------
    const vec3f surfPos = vec3f(optixGetWorldRayOrigin())
      + vec3f(optixGetRayTmax()) * vec3f(optixGetWorldRayDirection());

    // ------------------------------------------------------------------
    // trace shadow ray toward directional light
    // ------------------------------------------------------------------
    const vec3f L = optixLaunchParams.light.dir;

    vec3f lightVisibility = vec3f(0.f);
    {
      uint32_t u0, u1;
      packPointer( &lightVisibility, u0, u1 );
      optixTrace(optixLaunchParams.traversable,
                 surfPos + 1e-3f * worldNg,
                 L,
                 1e-3f,      // tmin
                 1e20f,      // tmax (directional light is infinitely far)
                 0.0f,       // rayTime
                 OptixVisibilityMask( 255 ),
                 // For shadow rays: skip any/closest hit shaders and
                 // terminate on first intersection. The miss shader is
                 // used to mark if the light was visible.
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT
                 | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                 | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                 SHADOW_RAY_TYPE,            // SBT offset
                 RAY_TYPE_COUNT,             // SBT stride
                 SHADOW_RAY_TYPE,            // missSBTIndex 
                 u0, u1 );
    }

    // ------------------------------------------------------------------
    // PBR lighting: UE5-style GGX microfacet model
    // ------------------------------------------------------------------
    const vec3f lightColor = optixLaunchParams.light.color
      * optixLaunchParams.light.intensity;

    // view direction: from hit point toward camera
    const vec3f V = -normalize(vec3f(optixGetWorldRayDirection()));

    const float NdotL = clamp(dot(shadingN, L), 1e-5f, 1.f);
    const float NdotV = clamp(dot(shadingN, V), 1e-5f, 1.f);
    const vec3f H = normalize(L + V);
    const float NdotH = clamp(dot(shadingN, H), 0.f, 1.f);
    const float VdotH = clamp(dot(V, H), 0.f, 1.f);

    // Disney roughness remapping
    const float a = roughness * roughness;

    // surface reflectance at normal incidence (F0)
    // dielectrics = 0.04, metals take on the albedo color
    vec3f F0 = vec3f(0.04f);
    F0 = F0 * (1.f - metallic) + albedo * metallic;

    // Cook-Torrance BRDF
    float D = D_GGX(NdotH, a);
    float G = Vis_SmithJointApprox(a, NdotV, NdotL);
    vec3f F = F_Schlick(VdotH, F0);

    vec3f specular = D * G * F;

    // diffuse: Lambertian scaled by energy conservation
    vec3f diffuse = (1.f - metallic) * albedo / M_PI;

    // final outgoing radiance with shadow (ambient not shadowed)
    vec3f Lo = (diffuse + specular) * NdotL * lightColor * lightVisibility
             + ao * albedo / M_PI;

    // write result
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = Lo;
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }

  //------------------------------------------------------------------------------
  // miss programs
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    // set to sky blue as background color
    prd = vec3f(0.2f, 0.3f, 0.9f);
  }

  extern "C" __global__ void __miss__shadow()
  {
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = vec3f(1.f);
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    vec3f pixelColorPRD = vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix+.5f,iy+.5f)
                       / vec2f(optixLaunchParams.frame.size));
    
    // generate ray direction
    vec3f rayDir = normalize(camera.direction
                             + (screen.x - 0.5f) * camera.horizontal
                             + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
               camera.position,
               rayDir,
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               RADIANCE_RAY_TYPE,            // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               RADIANCE_RAY_TYPE,            // missSBTIndex 
               u0, u1 );

    const int r = int(255.99f*pixelColorPRD.x);
    const int g = int(255.99f*pixelColorPRD.y);
    const int b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
  
} // ::osc
