#version 450 core
layout(std140, row_major, binding = 0)uniform transBuffer
{
	mat4 trans;
};
layout(location = 0)in vec2 position;
//layout(location = 1)in vec3 velocity;
out vec4 fragColor;
void main()
{
	gl_Position = trans * vec4(position, 0, 1);
	fragColor = vec4((1.f + tanh(position.x / 5))/2, (1.f + tanh(position.y / 5)) / 2, 1, 1);
}