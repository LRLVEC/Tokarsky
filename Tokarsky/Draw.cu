#define DefineDevice
#include "Define.h"


//To do: Construct again...

extern "C"
{
	__constant__ Parameters paras;
}

extern "C" __global__ void __raygen__RayAllocator()
{
	unsigned int index = optixGetLaunchIndex().x;
	RayData* rtData = (RayData*)optixGetSbtDataPointer();

	float ahh = 2 * 3.14159265358f * random(paras.pos + make_float2(index));
	float3 dd = { cos(ahh), sin(ahh), 0 };

	unsigned int pd0, pd1, depth(0);
	pP(paras.lines + 2 * index * paras.depth, pd0, pd1);
	optixTrace(paras.handle, make_float3(paras.pos, 0), dd,
		1e-6f, 1e16f,
		0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		RayRadiance,        // SBT offset
		RayCount,           // SBT stride
		RayRadiance,        // missSBTIndex
		pd0, pd1, depth);

}
extern "C" __global__ void __closesthit__Ahh()
{
	unsigned int depth(optixGetPayload_2());
	if (depth < paras.depth)
	{
		CloseHitData* closeHitData = (CloseHitData*)optixGetSbtDataPointer();
		int primIdx = optixGetPrimitiveIndex();
		float3 n = closeHitData->normals[primIdx];
		float3 rayDir(optixGetWorldRayDirection());
		float3 rayOrigin(optixGetWorldRayOrigin());

		unsigned int pd0 = optixGetPayload_0();
		unsigned int pd1 = optixGetPayload_1();
		float2* ptr((float2*)uP(pd0, pd1));
		float t(optixGetRayTmax());
		ptr[2 * depth] = { rayOrigin.x, rayOrigin.y };
		rayOrigin.x += t * rayDir.x;
		rayOrigin.y += t * rayDir.y;
		ptr[2 * depth + 1] = { rayOrigin.x, rayOrigin.y };
		++depth;
		rayDir -= 2 * dot(rayDir, n) * n;
		optixTrace(paras.handle, rayOrigin, rayDir,
			1e-6f, 1e16f,
			0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
			RayRadiance,        // SBT offset
			RayCount,           // SBT stride
			RayRadiance,        // missSBTIndex
			pd0, pd1, depth);
		optixSetPayload_2(depth);
	}
}
extern "C" __global__ void __miss__Ahh()
{
}
