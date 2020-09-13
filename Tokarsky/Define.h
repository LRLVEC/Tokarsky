#pragma once
#ifdef DefineDevice
#include <OptiX/_Define_7_Device.h>
#else
#endif
enum RayType
{
	RayRadiance = 0,
	RayCount
};
struct RayData
{
	float r, g, b;
};
struct CloseHitData
{
	float3* normals;
};
struct Parameters
{
	float2* lines;
	OptixTraversableHandle handle;
	float2 pos;
	unsigned int launchSize;
	unsigned int depth;
};