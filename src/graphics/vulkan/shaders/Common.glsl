// Shared definitions for the Vulkan renderer shaders.
// Must match VulkanPushConstants in VulkanRenderer.h

layout(push_constant) uniform PushConstants {
	mat4 transform;  // clip-space transform (projection * view, or screen-space ortho)
	vec4 viewZ;      // row of the view matrix that yields eye-space z (fog distance)
	vec4 fogColor;
	vec4 params;     // x: fog start, y: fog end, z: alpha reference (< 0 = no alpha test), w: fog enabled
	uvec4 ops;       // x: texture stage ops, 8 bits per stage: color op (low 4 bits), alpha op (high 4 bits)
} pc;
