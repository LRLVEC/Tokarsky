#include <cstdio>
#include <cstdlib>
#include <GL/_OpenGL.h>
#include <GL/_Window.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <OptiX/_OptiX_7.h>
#include "Define.h"
#include <_Time.h>
#include <_STL.h>

namespace CUDA
{
	namespace OptiX
	{
		struct PathTracing
		{
			Context context;
			OptixModuleCompileOptions moduleCompileOptions;
			OptixPipelineCompileOptions pipelineCompileOptions;
			ModuleManager mm;
			OptixProgramGroupOptions programGroupOptions;
			Program rayAllocator;
			Program miss;
			Program closestHit;
			OptixPipelineLinkOptions pipelineLinkOptions;
			Pipeline pip;
			SbtRecord<RayData> raygenData;
			SbtRecord<int> missData;
			SbtRecord<CloseHitData> hitData;
			Buffer raygenDataBuffer;
			Buffer missDataBuffer;
			Buffer hitDataBuffer;
			OptixShaderBindingTable sbt;
			Buffer linesBuffer;
			CUstream cuStream;
			Parameters paras;
			Buffer parasBuffer;
			STL box;
			Buffer vertices;
			Buffer normals;
			OptixBuildInput triangleBuildInput;
			OptixAccelBuildOptions accelOptions;
			Buffer GASOutput;
			OptixTraversableHandle GASHandle;
			PathTracing(OpenGL::SourceManager* _sourceManager, OpenGL::Buffer* _linesBuffer, unsigned int _launchSize, unsigned int _depth)
				:
				context(),
				moduleCompileOptions{
				OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
				OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
				OPTIX_COMPILE_DEBUG_LEVEL_FULL },
				pipelineCompileOptions{ false,
				OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
				3,2,OPTIX_EXCEPTION_FLAG_NONE,"paras", unsigned int(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE) },//OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE: new in OptiX7.1.0
				mm(&_sourceManager->folder, context, &moduleCompileOptions, &pipelineCompileOptions),
				programGroupOptions{},
				rayAllocator(Vector<String<char>>("__raygen__RayAllocator"), Program::RayGen, &programGroupOptions, context, &mm),
				miss(Vector<String<char>>("__miss__Ahh"), Program::Miss, &programGroupOptions, context, &mm),
				closestHit(Vector<String<char>>("__closesthit__Ahh"), Program::HitGroup, &programGroupOptions, context, &mm),
				pipelineLinkOptions{ _depth,OPTIX_COMPILE_DEBUG_LEVEL_FULL },//no overrideUsesMotionBlur in OptiX7.1.0
				pip(context, &pipelineCompileOptions, &pipelineLinkOptions, { rayAllocator ,closestHit, miss }),
				raygenDataBuffer(raygenData, false),
				missDataBuffer(missData, false),
				hitDataBuffer(hitData, false),
				sbt({}),
				linesBuffer(_linesBuffer->buffer),
				parasBuffer(paras, false),
				box(_sourceManager->folder.find("resources/room_0.stl").readSTL()),
				vertices(CUDA::Buffer::Device),
				normals(CUDA::Buffer::Device),
				triangleBuildInput({}),
				accelOptions({}),
				GASOutput(CUDA::Buffer::Device)
			{
				box.getVerticesRepeated();
				box.getNormals();
				box.printInfo(false);
				vertices.copy(box.verticesRepeated.data, sizeof(Math::vec3<float>)* box.verticesRepeated.length);
				normals.copy(box.normals.data, sizeof(Math::vec3<float>)* box.normals.length);
				uint32_t triangle_input_flags[1] =  // One per SBT record for this build input
				{
					OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
					//OPTIX_GEOMETRY_FLAG_NONE
				};

				triangleBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				triangleBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				//triangleBuildInput.triangleArray.vertexStrideInBytes = sizeof(Math::vec3<float>);
				triangleBuildInput.triangleArray.numVertices = box.verticesRepeated.length;
				triangleBuildInput.triangleArray.vertexBuffers = (CUdeviceptr*)&vertices.device;
				triangleBuildInput.triangleArray.flags = triangle_input_flags;
				triangleBuildInput.triangleArray.numSbtRecords = 1;
				triangleBuildInput.triangleArray.sbtIndexOffsetBuffer = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
				triangleBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
				triangleBuildInput.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;//new in OptiX7.1.0

				accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
				accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

				Buffer temp(Buffer::Device);
				Buffer compation(Buffer::Device);
				OptixAccelBufferSizes GASBufferSizes;
				optixAccelComputeMemoryUsage(context, &accelOptions, &triangleBuildInput, 1, &GASBufferSizes);
				temp.resize(GASBufferSizes.tempSizeInBytes);
				size_t compactedSizeOffset = ((GASBufferSizes.outputSizeInBytes + 7) / 8) * 8;
				compation.resize(compactedSizeOffset + 8);

				OptixAccelEmitDesc emitProperty = {};
				emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
				emitProperty.result = (CUdeviceptr)((char*)compation.device + compactedSizeOffset);

				optixAccelBuild(context, 0,
					&accelOptions, &triangleBuildInput, 1,// num build inputs, which is the num of vertexBuffers pointers
					temp, GASBufferSizes.tempSizeInBytes,
					compation, GASBufferSizes.outputSizeInBytes,
					&GASHandle, &emitProperty, 1);

				size_t compacted_gas_size;
				cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost);
				::printf("Compatcion: %u to %u\n", GASBufferSizes.outputSizeInBytes, compacted_gas_size);
				if (compacted_gas_size < GASBufferSizes.outputSizeInBytes)
				{
					GASOutput.resize(compacted_gas_size);
					// use handle as input and output
					optixAccelCompact(context, 0, GASHandle, GASOutput, compacted_gas_size, &GASHandle);
				}
				else GASOutput.copy(compation);
				paras.handle = GASHandle;
				paras.depth = _depth;
				/*OptixStackSizes stackSizes = { 0 };
				optixUtilAccumulateStackSizes(programGroups[0], &stackSizes);

				uint32_t max_trace_depth = 1;
				uint32_t max_cc_depth = 0;
				uint32_t max_dc_depth = 0;
				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				optixUtilComputeStackSizes(&stackSizes,
					max_trace_depth, max_cc_depth, max_dc_depth,
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state,
					&continuation_stack_size
				);
				optixPipelineSetStackSize(pipeline,
					direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state,
					continuation_stack_size, 3);*/
				optixSbtRecordPackHeader(rayAllocator, &raygenData);
				raygenData.data = { 0.462f, 0.725f, 0.f };
				raygenDataBuffer.copy(raygenData);
				optixSbtRecordPackHeader(miss, &missData);
				missDataBuffer.copy(missData);
				optixSbtRecordPackHeader(closestHit, &hitData);
				hitData.data.normals = (float3*)normals.device;
				hitDataBuffer.copy(hitData);

				sbt.raygenRecord = raygenDataBuffer;
				sbt.missRecordBase = missDataBuffer;
				sbt.missRecordStrideInBytes = sizeof(SbtRecord<int>);
				sbt.missRecordCount = 1;
				sbt.hitgroupRecordBase = hitDataBuffer;
				sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<CloseHitData>);
				sbt.hitgroupRecordCount = 1;

				cudaStreamCreate(&cuStream);
			}
			void run()
			{
				linesBuffer.map();
				cudaDeviceSynchronize();
				optixLaunch(pip, cuStream, parasBuffer, sizeof(Parameters), &sbt, paras.launchSize, 1, 1);
				cudaDeviceSynchronize();
				linesBuffer.unmap();
				cudaDeviceSynchronize();
			}
			void resize(unsigned int _launchSize, GLuint _gl)
			{
				linesBuffer.resize(_gl);
				linesBuffer.map();
				paras.lines = (float2*)linesBuffer.device;
				paras.launchSize = _launchSize;
				parasBuffer.copy(paras);
				linesBuffer.unmap();
			}
			void refreshSourcePos(float2 _pos)
			{
				paras.pos = _pos;
				parasBuffer.copy(paras);
			}
		};
	}
}
namespace OpenGL
{
	struct PathTracing :OpenGL
	{
		struct Border :Buffer::Data
		{
			float data[54] =
			{
				0, 0, 1, 0, 1, -1,
				2, -1, 2, 0, 3, 1,
				2, 1, 2, 2, 1, 2,
				1, 4, 2, 4, 2, 5,
				3, 5, 3, 6, 2, 6,
				1, 7, 1, 6, 0, 6,
				0, 5, -1, 5, 0, 4,
				0, 3, -1, 3, 0, 2,
				0, 1, -1, 1, 0, 0,
			};
			virtual void* pointer()override
			{
				return data;
			}
			virtual unsigned int size()override
			{
				return sizeof(data);
			}
		};
		struct Test :Buffer::Data
		{
			float data[4] = { 0, 0, 1, 1 };
			virtual void* pointer()override
			{
				return data;
			}
			virtual unsigned int size()override
			{
				return sizeof(data);
			}
		};
		struct Renderer :Program
		{
			Buffer transBuffer;
			BufferConfig transUniform;
			BufferConfig borderArray;
			BufferConfig testArray;
			BufferConfig linesArray;
			VertexAttrib borderPositions;
			VertexAttrib testPositions;
			VertexArrayBuffer vao1;

			Renderer(SourceManager* _sm, Buffer* _borderBuffer, Transform* _trans, Buffer* _linesBuffer, Buffer* _testBuffer)
				:
				Program(_sm, "Renderer", Vector<VertexAttrib*>{&borderPositions/*, & velocities*/}),
				transBuffer(&_trans->bufferData),
				transUniform(&transBuffer, UniformBuffer, 0),
				borderArray(_borderBuffer, ArrayBuffer),
				testArray(_testBuffer, ArrayBuffer),
				linesArray(_linesBuffer, ArrayBuffer),
				borderPositions(&borderArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(float) * 2, 0, 0),
				testPositions(&linesArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(float) * 2, 0, 0),
				vao1(Vector<VertexAttrib*>{&testPositions/*, & velocities*/})
			{
				init();
				vao1.init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				vao.bind();
				glLineWidth(2);
				glDrawArrays(GL_LINE_STRIP, 0, 27);
				vao1.bind();
				glLineWidth(1);
				glDrawArrays(GL_LINES, 0, linesArray.buffer->data->size() / sizeof(float2));
			}
		};

		struct Lines :Buffer::Data
		{
			unsigned int num;
			Lines(unsigned int _num)
				:
				num(_num)
			{
			}
			virtual void* pointer()override
			{
				return nullptr;
			}
			virtual unsigned int size()override
			{
				return num * 2 * sizeof(float2);
			}
		};


		SourceManager sm;
		//OptiXDefautRenderer renderer;
		Border border;
		Test test;
		Lines lines;
		Buffer borderBuffer;
		Buffer testBuffer;
		Buffer linesBuffer;
		Transform trans;
		Transform::Key moveSource;
		Renderer renderer;
		unsigned int launchSize;
		unsigned int depth;
		float2 sourcePos;
		CUDA::OptiX::PathTracing pathTracer;
		FrameScale size;
		bool frameSizeChanged;
		PathTracing(FrameScale _size, unsigned int _launchSize, unsigned int _depth)
			:
			sm(),
			border(),
			test(),
			lines(_launchSize* _depth),
			borderBuffer(&border),
			testBuffer(&test),
			linesBuffer(&lines),
			trans({ {80.0,0.1,800},{0.08,0.9,0.01},{0.1},500.0 }),
			moveSource({ 0.01 }),
			renderer(&sm, &borderBuffer, &trans, &linesBuffer, &testBuffer),
			launchSize(_launchSize),
			depth(_depth),
			sourcePos{ 0.5f, 0.5f },
			pathTracer(&sm, &linesBuffer, _launchSize, _depth),
			size(_size),
			frameSizeChanged(false)
		{
			trans.init(_size);
		}
		virtual void init(FrameScale const& _size) override
		{
			glViewport(0, 0, _size.w, _size.h);
			glEnable(GL_DEPTH_TEST);
			trans.init(_size);
			renderer.transUniform.dataInit();
			renderer.borderArray.dataInit();
			renderer.testArray.dataInit();
			renderer.linesArray.dataInit();
			pathTracer.paras.pos = sourcePos;
			pathTracer.resize(launchSize, linesBuffer.buffer);
			glUseProgram(renderer.program);
		}
		virtual void run() override
		{
			changeFrameSize();
			trans.operate();
			if (trans.updated)
			{
				renderer.transUniform.refreshData();
				trans.updated = false;
			}

			Math::vec2<double> offset(moveSource.operate());
			sourcePos.x += offset[0];
			sourcePos.y += offset[1];
			pathTracer.refreshSourcePos(sourcePos);

			pathTracer.run();
			renderer.run();
		}
		void terminate()
		{
		}
		void changeFrameSize()
		{
			if (frameSizeChanged)
			{
				glFinish();
				trans.resize(size.w, size.h);
				glViewport(0, 0, size.w, size.h);
				frameSizeChanged = false;
			}
		}
		virtual void frameSize(int _w, int _h)override
		{
			if (size.w != _w || size.h != _h)
			{
				frameSizeChanged = true;
				size.w = _w;
				size.h = _h;
			}
		}
		virtual void framePos(int, int) override {}
		virtual void frameFocus(int) override {}
		virtual void mouseButton(int _button, int _action, int _mods)override
		{
			switch (_button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
			case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
			case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y)override
		{
			trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(double _x, double _y)override
		{
			if (_y != 0.0)
				trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			{
				switch (_key)
				{
				case GLFW_KEY_ESCAPE:if (_action == GLFW_PRESS)glfwSetWindowShouldClose(_window, true); break;
				case GLFW_KEY_LEFT:moveSource.refresh(0, _action); break;
				case GLFW_KEY_RIGHT:moveSource.refresh(1, _action); break;
				case GLFW_KEY_UP:moveSource.refresh(2, _action); break;
				case GLFW_KEY_DOWN:moveSource.refresh(3, _action); break;
				case GLFW_KEY_A:trans.key.refresh(0, _action); break;
				case GLFW_KEY_D:trans.key.refresh(1, _action); break;
				case GLFW_KEY_W:trans.key.refresh(2, _action); break;
				case GLFW_KEY_S:trans.key.refresh(3, _action); break;
					/*	case GLFW_KEY_UP:monteCarlo.trans.persp.increaseV(0.02); break;
						case GLFW_KEY_DOWN:monteCarlo.trans.persp.increaseV(-0.02); break;
						case GLFW_KEY_RIGHT:monteCarlo.trans.persp.increaseD(0.01); break;
						case GLFW_KEY_LEFT:monteCarlo.trans.persp.increaseD(-0.01); break;*/
				}
			}
		}
	};
}

int main()
{
	OpenGL::OpenGLInit(4, 5);
	Window::Window::Data winPara
	{
		"PathTracer",
		{
			{1080,1920},
			true,false
		}
	};
	Window::WindowManager wm(winPara);
	OpenGL::PathTracing pathTracer(winPara.size.size, 10000, 10);
	wm.init(0, &pathTracer);
	glfwSwapInterval(1);
	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		fps.refresh();
		fps.printFPSAndFrameTime(2, 3);
		//wm.windows[0].data.setTitle(fps.str);
	}
	return 1;
}