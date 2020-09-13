#include <cstdio>
#include <GL/_Window.h>
#include <_Math.h>
#include <_Time.h>
#include <random>

namespace OpenGL
{
	struct RenderLine :OpenGL
	{
		struct Particles
		{
			struct Particle
			{
				Math::vec3<float>position;
				float mass;
				Math::vec4<float>velocity;
			};
			Vector<Particle>particles;
			std::mt19937 mt;
			std::uniform_real_distribution<float>randReal;
			unsigned int num;
			Particles() = delete;
			Particles(unsigned int _num)
				:
				num(_num),
				randReal(0, 1)
			{
			}
			Particle flatGalaxyParticles()
			{
				float r(100 * randReal(mt) + 0.1);
				float phi(2 * Math::Pi * randReal(mt));
				r = pow(r, 0.5);
				float vk(2.0f);
				float rn(0.3);
				return
				{
					{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),0,vk * cos(phi) / powf(r,rn)},
				};
			}
			Particle sphereGalaxyParticles()
			{
				float r(pow(100.0f * randReal(mt) + 0.1f, 1.0 / 3));
				float theta(2.0f * acos(randReal(mt)));
				float phi(2 * Math::Pi * randReal(mt));
				float vk(1.7f);
				float rn(0.5);
				return
				{
					{r * cos(phi) * sin(theta),r * sin(phi) * sin(theta),r * cos(theta)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
				};
			}
			void randomGalaxy()
			{
				unsigned int _num(num - 1);
				while (_num--)
					particles.pushBack(flatGalaxyParticles());
				particles.pushBack
				(
					{
						{0,0,0},
						8000,
						{0,0,0},
					}
				);
			}
		};
		struct ParticlesData :Buffer::Data
		{
			Particles* particles;
			ParticlesData(Particles* _particles)
				:
				Data(DynamicDraw),
				particles(_particles)
			{
			}
			virtual void* pointer()override
			{
				return particles->particles.data;
			}
			virtual unsigned int size()override
			{
				return sizeof(Particles::Particle) * (particles->particles.length);
			}
		};

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


		struct Renderer :Program
		{
			Buffer transBuffer;
			BufferConfig transUniform;
			BufferConfig borderArray;
			VertexAttrib positions;
			//VertexAttrib velocities;

			Renderer(SourceManager* _sm, Buffer* _particlesBuffer, Transform* _trans)
				:
				Program(_sm, "Renderer", Vector<VertexAttrib*>{&positions/*, & velocities*/}),
				transBuffer(&_trans->bufferData),
				transUniform(&transBuffer, UniformBuffer, 0),
				borderArray(_particlesBuffer, ArrayBuffer),
				positions(&borderArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(float)*2, 0, 0)
				//velocities(&particlesArray, 1, VertexAttrib::three, VertexAttrib::Float, false, sizeof(Particles::Particle), 16, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glDrawArrays(GL_LINE_STRIP, 0, 27);
			}
		};

		SourceManager sm;
		Border border;
		Buffer borderBuffer;
		Transform trans;
		Renderer renderer;

		RenderLine(unsigned int _groups)
			:
			sm(),
			border(),
			borderBuffer(&border),
			trans({ {80.0,0.1,800},{0.08,0.9,0.01},{0.1},500.0 }),
			renderer(&sm, &borderBuffer, &trans)
		{
			//particles.randomGalaxy();
		}
		virtual void init(FrameScale const& _size)override
		{
			glViewport(0, 0, _size.w, _size.h);
			//glPointSize(2);
			glLineWidth(2);
			glEnable(GL_DEPTH_TEST);
			trans.init(_size);
			renderer.transUniform.dataInit();
			renderer.borderArray.dataInit();
		}
		virtual void run()override
		{
			trans.operate();
			if (trans.updated)
			{
				renderer.transUniform.refreshData();
				trans.updated = false;
			}
			renderer.use();
			renderer.run();
		}
		virtual void frameSize(int _w, int _h) override
		{
			trans.resize(_w, _h);
			glViewport(0, 0, _w, _h);
		}
		virtual void framePos(int, int) override
		{
		}
		virtual void frameFocus(int) override
		{
		}
		virtual void mouseButton(int _button, int _action, int _mods) override
		{
			switch (_button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
			case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
			case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y) override
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
			switch (_key)
			{
			case GLFW_KEY_ESCAPE:
				if (_action == GLFW_PRESS)
					glfwSetWindowShouldClose(_window, true);
				break;
			case GLFW_KEY_A:trans.key.refresh(0, _action); break;
			case GLFW_KEY_D:trans.key.refresh(1, _action); break;
			case GLFW_KEY_W:trans.key.refresh(2, _action); break;
			case GLFW_KEY_S:trans.key.refresh(3, _action); break;
			}
		}
	};
}


int main()
{
	OpenGL::OpenGLInit init(4, 5);
	Window::Window::Data winParameters
	{
		"RenderLine",
		{
			{1920,1080},
			true,false
		}
	};
	Window::WindowManager wm(winParameters);
	OpenGL::RenderLine nBody(20);
	wm.init(0, &nBody);
	init.printRenderer();
	glfwSwapInterval(1);
	FPS fps;
	fps.refresh();
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		fps.refresh();
		::printf("\r%.2lf    ", fps.fps);
		//fps.printFPS(1);
	}
	return 0;
}