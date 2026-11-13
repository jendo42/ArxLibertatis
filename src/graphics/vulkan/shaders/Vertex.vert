#version 450
#extension GL_GOOGLE_include_directive : require

// World-space vertices with one (SMY_VERTEX) or three (SMY_VERTEX3) texture coordinate sets

#include "Common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord0;
#ifdef THREE_TEXCOORDS
layout(location = 3) in vec2 inTexCoord1;
layout(location = 4) in vec2 inTexCoord2;
#endif

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outTexCoord0;
layout(location = 2) out vec2 outTexCoord1;
layout(location = 3) out vec2 outTexCoord2;
layout(location = 4) out float outFogDepth;

void main() {
	vec4 position = pc.transform * vec4(inPosition, 1.0);
	// OpenGL depth range [-1, 1] -> Vulkan [0, 1]
	position.z = (position.z + position.w) * 0.5;
	gl_Position = position;
	outColor = inColor;
	outTexCoord0 = inTexCoord0;
	#ifdef THREE_TEXCOORDS
	outTexCoord1 = inTexCoord1;
	outTexCoord2 = inTexCoord2;
	#else
	outTexCoord1 = vec2(0.0);
	outTexCoord2 = vec2(0.0);
	#endif
	outFogDepth = abs(dot(pc.viewZ, vec4(inPosition, 1.0)));
}
