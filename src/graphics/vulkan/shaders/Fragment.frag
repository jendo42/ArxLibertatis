#version 450
#extension GL_GOOGLE_include_directive : require

// Emulates the fixed-function texture stages, alpha test and linear fog
// used by the OpenGL renderer.

#include "Common.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[3];

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inTexCoord0;
layout(location = 2) in vec2 inTexCoord1;
layout(location = 3) in vec2 inTexCoord2;
layout(location = 4) in float inFogDepth;

layout(location = 0) out vec4 outColor;

// Must match TextureStage::TextureOp
const uint OpDisable = 0u;
const uint OpSelectArg1 = 1u;
const uint OpModulate = 2u;
const uint OpModulate2X = 3u;
const uint OpModulate4X = 4u;

vec3 combine(uint op, vec3 previous, vec3 texel) {
	switch(op) {
		case OpSelectArg1: return texel;
		case OpModulate:   return previous * texel;
		case OpModulate2X: return min(previous * texel * 2.0, vec3(1.0));
		case OpModulate4X: return min(previous * texel * 4.0, vec3(1.0));
		default:           return previous;
	}
}

float combine(uint op, float previous, float texel) {
	switch(op) {
		case OpSelectArg1: return texel;
		case OpModulate:   return previous * texel;
		case OpModulate2X: return min(previous * texel * 2.0, 1.0);
		case OpModulate4X: return min(previous * texel * 4.0, 1.0);
		default:           return previous;
	}
}

vec4 applyStage(uint stage, vec4 previous, vec2 texCoord) {
	uint ops = (pc.ops.x >> (stage * 8u)) & 0xffu;
	if(ops == 0u) {
		return previous;
	}
	vec4 texel;
	switch(stage) {
		case 0u: texel = texture(textures[0], texCoord); break;
		case 1u: texel = texture(textures[1], texCoord); break;
		default: texel = texture(textures[2], texCoord); break;
	}
	return vec4(combine(ops & 0xfu, previous.rgb, texel.rgb), combine(ops >> 4u, previous.a, texel.a));
}

void main() {

	vec4 color = inColor;
	color = applyStage(0u, color, inTexCoord0);
	color = applyStage(1u, color, inTexCoord1);
	color = applyStage(2u, color, inTexCoord2);

	if(pc.params.z >= 0.0 && !(color.a > pc.params.z)) {
		discard;
	}

	if(pc.params.w != 0.0) {
		float range = pc.params.y - pc.params.x;
		float f = (range != 0.0) ? clamp((pc.params.y - inFogDepth) / range, 0.0, 1.0) : 1.0;
		color.rgb = mix(pc.fogColor.rgb, color.rgb, f);
	}

	outColor = color;
}
