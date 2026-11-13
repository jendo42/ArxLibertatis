#version 450
#extension GL_GOOGLE_include_directive : require

// Pre-transformed (screen-space) vertices: xyz pre-multiplied by w as for OpenGL

#include "Common.glsl"

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord0;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outTexCoord0;
layout(location = 2) out vec2 outTexCoord1;
layout(location = 3) out vec2 outTexCoord2;
layout(location = 4) out float outFogDepth;

void main() {
	vec4 position = pc.transform * inPosition;
	// OpenGL depth range [-1, 1] -> Vulkan [0, 1]
	position.z = (position.z + position.w) * 0.5;
	gl_Position = position;
	outColor = inColor;
	outTexCoord0 = inTexCoord0;
	outTexCoord1 = vec2(0.0);
	outTexCoord2 = vec2(0.0);
	outFogDepth = inPosition.w;
}
