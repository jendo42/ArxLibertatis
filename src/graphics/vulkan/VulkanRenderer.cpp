/*
 * Copyright 2026 Arx Libertatis Team (see the AUTHORS file)
 *
 * This file is part of Arx Libertatis.
 *
 * Arx Libertatis is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Arx Libertatis is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Arx Libertatis.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "graphics/vulkan/VulkanRenderer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string_view>

#include <glm/gtc/matrix_transform.hpp>

#include "core/Config.h"
#include "graphics/vulkan/VulkanTexture.h"
#include "graphics/vulkan/VulkanTextureStage.h"
#include "graphics/vulkan/VulkanVertexBuffer.h"
#include "gui/Credits.h"
#include "io/log/Logger.h"
#include "platform/CrashHandler.h"

// SPIR-V generated at build time by glslangValidator
#include "vulkan-shaders/TexturedVertex.vert.h"
#include "vulkan-shaders/Vertex.vert.h"
#include "vulkan-shaders/Vertex3.vert.h"
#include "vulkan-shaders/Fragment.frag.h"

namespace {

const char * const vkValidationLayer = "VK_LAYER_KHRONOS_validation";
const char * const vkPortabilitySubsetExtension = "VK_KHR_portability_subset";
const char * const vkPortabilityEnumerationExtension = "VK_KHR_portability_enumeration";

bool vkIsDebugEnabled() {
	const char * env = std::getenv("ARX_VULKAN_DEBUG");
	return env && *env && std::string_view(env) != "0";
}

VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT type,
                                             const VkDebugUtilsMessengerCallbackDataEXT * data,
                                             void * userData) {
	ARX_UNUSED(type), ARX_UNUSED(userData);
	if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		LogError << "Vulkan: " << data->pMessage;
	} else if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		LogWarning << "Vulkan: " << data->pMessage;
	} else {
		LogInfo << "Vulkan: " << data->pMessage;
	}
	return VK_FALSE;
}

std::string vkVersionString(uint32_t version) {
	std::ostringstream oss;
	oss << VK_API_VERSION_MAJOR(version) << '.' << VK_API_VERSION_MINOR(version) << '.'
	    << VK_API_VERSION_PATCH(version);
	return oss.str();
}

bool vkHasExtension(const std::vector<VkExtensionProperties> & extensions, const char * name) {
	return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties & e) {
		return std::strcmp(e.extensionName, name) == 0;
	});
}

const VkBlendFactor arxToVkBlendFactor[] = {
	VK_BLEND_FACTOR_ZERO,                 // BlendZero
	VK_BLEND_FACTOR_ONE,                  // BlendOne
	VK_BLEND_FACTOR_SRC_COLOR,            // BlendSrcColor
	VK_BLEND_FACTOR_SRC_ALPHA,            // BlendSrcAlpha
	VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,  // BlendInvSrcColor
	VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // BlendInvSrcAlpha
	VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,   // BlendSrcAlphaSaturate
	VK_BLEND_FACTOR_DST_COLOR,            // BlendDstColor
	VK_BLEND_FACTOR_DST_ALPHA,            // BlendDstAlpha
	VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,  // BlendInvDstColor
	VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,  // BlendInvDstAlpha
};

const VkSamplerAddressMode arxToVkWrapMode[] = {
	VK_SAMPLER_ADDRESS_MODE_REPEAT,          // WrapRepeat
	VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, // WrapMirror
	VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   // WrapClamp
};

const VkFilter arxToVkFilter[] = {
	VK_FILTER_NEAREST, // FilterNearest
	VK_FILTER_LINEAR,  // FilterLinear
};

constexpr VkDeviceSize vkMinChunkSize = 4 * 1024 * 1024;
constexpr uint32_t vkDescriptorPoolSize = 1024;

bool vkIsPowerOfTwo(int value) {
	return value > 0 && (value & (value - 1)) == 0;
}

} // anonymous namespace

bool VulkanRenderer::DescriptorKey::operator==(const DescriptorKey & o) const {
	return std::memcmp(this, &o, sizeof(*this)) == 0;
}

size_t VulkanRenderer::DescriptorKeyHash::operator()(const DescriptorKey & key) const {
	size_t hash = 0;
	for(size_t i = 0; i < TextureStageCount; i++) {
		hash = hash * 31 + std::hash<u64>()(u64(reinterpret_cast<uintptr_t>(key.views[i])));
		hash = hash * 31 + std::hash<u64>()(u64(reinterpret_cast<uintptr_t>(key.samplers[i])));
	}
	return hash;
}

VulkanRenderer::VulkanRenderer(VulkanWindow & window)
	: m_window(window)
	, m_instance(VK_NULL_HANDLE)
	, m_debugMessenger(VK_NULL_HANDLE)
	, m_surface(VK_NULL_HANDLE)
	, m_physicalDevice(VK_NULL_HANDLE)
	, m_properties()
	, m_device(VK_NULL_HANDLE)
	, m_queueFamily(0)
	, m_queue(VK_NULL_HANDLE)
	, m_allocator(VK_NULL_HANDLE)
	, m_hasFillModeNonSolid(false)
	, m_hasAnisotropy(false)
	, m_hasDebugUtils(false)
	, m_hasPortabilitySubset(false)
	, m_hasSampleShading(false)
	, m_maxSamples(VK_SAMPLE_COUNT_1_BIT)
	, m_samples(VK_SAMPLE_COUNT_1_BIT)
	, m_antialiasingEnabled(false)
	, m_depthFormat(VK_FORMAT_UNDEFINED)
	, m_clearRenderPass(VK_NULL_HANDLE)
	, m_loadRenderPass(VK_NULL_HANDLE)
	, m_descriptorSetLayout(VK_NULL_HANDLE)
	, m_pipelineLayout(VK_NULL_HANDLE)
	, m_pipelineCache(VK_NULL_HANDLE)
	, m_vertexShaders()
	, m_fragmentShader(VK_NULL_HANDLE)
	, m_dummyImage(VK_NULL_HANDLE)
	, m_dummyView(VK_NULL_HANDLE)
	, m_dummyAllocation(VK_NULL_HANDLE)
	, m_swapchainDirty(true)
	, m_presentMode(VK_PRESENT_MODE_FIFO_KHR)
	, m_vsync(1)
	, m_frameNumber(1)
	, m_completedFrame(0)
	, m_imageIndex(0)
	, m_imageAcquired(false)
	, m_acquireWaited(false)
	, m_inRenderPass(false)
	, m_skipFrame(false)
	, m_renderPassCount(0)
	, m_queueStop(false)
	, m_queueBusy(false)
	, m_submittedFrame(0)
	, m_presentOutOfDate(false)
	, m_readbackBuffer(VK_NULL_HANDLE)
	, m_readbackAllocation(VK_NULL_HANDLE)
	, m_readbackSize(0)
	, m_projection(1.f)
	, m_view(1.f)
	, m_viewProjection(1.f)
	, m_screenTransform(1.f)
	, m_fogColor(Color4f::black)
	, m_fogStart(0.f)
	, m_fogEnd(1.f)
	, m_fillMode(FillSolid)
	, m_maxAnisotropy(1.f)
	, m_maxSupportedAnisotropy(1.f)
	, m_boundPipeline(VK_NULL_HANDLE)
	, m_boundDescriptorSet(VK_NULL_HANDLE)
	, m_boundVertexBuffer(VK_NULL_HANDLE)
	, m_pushConstants()
	, m_pushConstantsValid(false)
	, m_dynamicStateValid(false)
	, m_boundDepthOffset(0)
	, m_pipelineType(VulkanTexturedVertex)
	, m_pipelineTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
	, m_pipelineFillMode(FillSolid)
	, m_pipelineAlphaTest(AlphaTestNone)
	, m_textureState()
	, m_textureStateValid(false)
	, m_textureDescriptorSet(VK_NULL_HANDLE)
	, m_textureOps(0)
{ }

VulkanRenderer::~VulkanRenderer() {

	stopQueueThread();

	if(m_device) {
		vkDeviceWaitIdle(m_device);
	}

	if(isInitialized()) {
		onRendererShutdown();
	}

	m_TextureStages.clear();

	destroy();
}

// Initialization

bool VulkanRenderer::create() {

	if(!createInstance()) {
		return false;
	}

	m_surface = m_window.createVulkanSurface(m_instance);
	if(m_surface == VK_NULL_HANDLE) {
		LogError << "Could not create Vulkan surface";
		return false;
	}

	if(!selectPhysicalDevice() || !createDevice()) {
		return false;
	}

	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;
	allocatorInfo.physicalDevice = m_physicalDevice;
	allocatorInfo.device = m_device;
	allocatorInfo.instance = m_instance;
	if(vmaCreateAllocator(&allocatorInfo, &m_allocator) != VK_SUCCESS) {
		LogError << "Could not create Vulkan memory allocator";
		return false;
	}

	m_samples = desiredSampleCount();
	if(!createRenderPasses() || !createPipelineLayout() || !createShaders()) {
		return false;
	}

	createFrames();
	prepareFrame();
	createDummyTexture();
	
	const char * syncSubmit = std::getenv("ARX_VULKAN_SYNC_SUBMIT");
	if(!syncSubmit || !*syncSubmit || std::string_view(syncSubmit) == "0") {
		m_queueThread = std::thread(&VulkanRenderer::queueThread, this);
	}

	return true;
}

bool VulkanRenderer::createInstance() {

	uint32_t count = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
	std::vector<VkExtensionProperties> available(count);
	vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

	std::vector<const char *> extensions;
	if(!m_window.getVulkanInstanceExtensions(extensions)) {
		LogError << "Could not get Vulkan instance extensions for the window";
		return false;
	}

	VkInstanceCreateFlags flags = 0;
	if(vkHasExtension(available, vkPortabilityEnumerationExtension)) {
		extensions.push_back(vkPortabilityEnumerationExtension);
		flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}

	std::vector<const char *> layers;
	bool debug = vkIsDebugEnabled();
	if(debug) {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
		for(const VkLayerProperties & layer : availableLayers) {
			if(std::strcmp(layer.layerName, vkValidationLayer) == 0) {
				layers.push_back(vkValidationLayer);
			}
		}
		if(layers.empty()) {
			LogWarning << "Vulkan validation layer not available";
		}
		if(vkHasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			m_hasDebugUtils = true;
		}
	}

	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Arx Libertatis";
	appInfo.pEngineName = "Arx Libertatis";
	appInfo.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.flags = flags;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = uint32_t(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledLayerCount = uint32_t(layers.size());
	createInfo.ppEnabledLayerNames = layers.data();

	VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan instance: " << int(result);
		m_instance = VK_NULL_HANDLE;
		return false;
	}

	if(m_hasDebugUtils) {
		auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
		if(createMessenger) {
			VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {};
			messengerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			                                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			                            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			                            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			messengerInfo.pfnUserCallback = vkDebugCallback;
			createMessenger(m_instance, &messengerInfo, nullptr, &m_debugMessenger);
		}
	}

	return true;
}

bool VulkanRenderer::selectPhysicalDevice() {

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

	int bestScore = 0;

	for(VkPhysicalDevice device : devices) {

		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		if(properties.apiVersion < VK_API_VERSION_1_1) {
			continue;
		}

		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
		if(!vkHasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
			continue;
		}

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
		if(formatCount == 0) {
			continue;
		}

		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
		std::vector<VkQueueFamilyProperties> families(familyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
		uint32_t family = familyCount;
		for(uint32_t i = 0; i < familyCount; i++) {
			VkBool32 present = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &present);
			if((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
				family = i;
				break;
			}
		}
		if(family == familyCount) {
			continue;
		}

		int score = 1;
		if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score = 4;
		} else if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
			score = 3;
		} else if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
			score = 2;
		}

		if(score > bestScore) {
			bestScore = score;
			m_physicalDevice = device;
			m_properties = properties;
			m_queueFamily = family;
			m_hasPortabilitySubset = vkHasExtension(extensions, vkPortabilitySubsetExtension);
		}

	}

	if(m_physicalDevice == VK_NULL_HANDLE) {
		LogError << "No suitable Vulkan device found";
		return false;
	}

	// Choose the surface format - keep it stable across swapchain recreation
	uint32_t formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, formats.data());
	m_swapchain.format = VK_FORMAT_UNDEFINED;
	for(const VkSurfaceFormatKHR & format : formats) {
		if((format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_R8G8B8A8_UNORM)
		   && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			m_swapchain.format = format.format;
			break;
		}
	}
	if(m_swapchain.format == VK_FORMAT_UNDEFINED) {
		m_swapchain.format = (formats[0].format == VK_FORMAT_UNDEFINED) ? VK_FORMAT_B8G8R8A8_UNORM
		                                                                : formats[0].format;
	}

	// Choose the depth format
	const VkFormat depthFormats[] = { VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_X8_D24_UNORM_PACK32,
	                                  VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM };
	for(VkFormat format : depthFormats) {
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &properties);
		if(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			m_depthFormat = format;
			break;
		}
	}
	if(m_depthFormat == VK_FORMAT_UNDEFINED) {
		LogError << "No supported Vulkan depth format";
		return false;
	}

	return true;
}

bool VulkanRenderer::createDevice() {

	VkPhysicalDeviceFeatures supported;
	vkGetPhysicalDeviceFeatures(m_physicalDevice, &supported);

	VkPhysicalDeviceFeatures features = {};
	if(supported.samplerAnisotropy) {
		features.samplerAnisotropy = VK_TRUE;
		m_hasAnisotropy = true;
	}
	if(supported.fillModeNonSolid) {
		features.fillModeNonSolid = VK_TRUE;
		m_hasFillModeNonSolid = true;
	}
	if(supported.sampleRateShading) {
		features.sampleRateShading = VK_TRUE;
		m_hasSampleShading = true;
	}
	
	VkSampleCountFlags sampleCounts = m_properties.limits.framebufferColorSampleCounts
	                                 & m_properties.limits.framebufferDepthSampleCounts;
	for(VkSampleCountFlagBits count : { VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT }) {
		if(sampleCounts & count) {
			m_maxSamples = count;
			break;
		}
	}

	float priority = 1.f;
	VkDeviceQueueCreateInfo queueInfo = {};
	queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueInfo.queueFamilyIndex = m_queueFamily;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;

	std::vector<const char *> extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	if(m_hasPortabilitySubset) {
		extensions.push_back(vkPortabilitySubsetExtension);
	}

	VkDeviceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueInfo;
	createInfo.enabledExtensionCount = uint32_t(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.pEnabledFeatures = &features;

	VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan device: " << int(result);
		m_device = VK_NULL_HANDLE;
		return false;
	}

	vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

	m_maxSupportedAnisotropy = m_hasAnisotropy ? m_properties.limits.maxSamplerAnisotropy : 1.f;

	return true;
}

bool VulkanRenderer::createRenderPasses() {

	const bool msaa = (m_samples != VK_SAMPLE_COUNT_1_BIT);

	for(int load = 0; load < 2; load++) {

		VkAttachmentDescription attachments[3] = {};

		// Color: the swapchain image, or a multisampled buffer that is resolved to it
		attachments[0].format = m_swapchain.format;
		attachments[0].samples = m_samples;
		attachments[0].loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		if(msaa) {
			attachments[0].initialLayout = load ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		} else {
			attachments[0].initialLayout = load ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		}

		attachments[1].format = m_depthFormat;
		attachments[1].samples = m_samples;
		attachments[1].loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = load ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		                                    : VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// Resolve target: the swapchain image, fully overwritten at the end of the render pass
		attachments[2].format = m_swapchain.format;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthRef = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
		VkAttachmentReference resolveRef = { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorRef;
		subpass.pResolveAttachments = msaa ? &resolveRef : nullptr;
		subpass.pDepthStencilAttachment = &depthRef;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		                          | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
		                          | VK_PIPELINE_STAGE_TRANSFER_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
		                           | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
		                           | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
		                           | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		createInfo.attachmentCount = msaa ? 3 : 2;
		createInfo.pAttachments = attachments;
		createInfo.subpassCount = 1;
		createInfo.pSubpasses = &subpass;
		createInfo.dependencyCount = 1;
		createInfo.pDependencies = &dependency;

		VkRenderPass & pass = load ? m_loadRenderPass : m_clearRenderPass;
		if(vkCreateRenderPass(m_device, &createInfo, nullptr, &pass) != VK_SUCCESS) {
			LogError << "Could not create Vulkan render pass";
			pass = VK_NULL_HANDLE;
			return false;
		}

	}

	return true;
}

void VulkanRenderer::destroyRenderPasses() {
	if(m_clearRenderPass) {
		vkDestroyRenderPass(m_device, m_clearRenderPass, nullptr);
		m_clearRenderPass = VK_NULL_HANDLE;
	}
	if(m_loadRenderPass) {
		vkDestroyRenderPass(m_device, m_loadRenderPass, nullptr);
		m_loadRenderPass = VK_NULL_HANDLE;
	}
}

bool VulkanRenderer::createPipelineLayout() {

	VkDescriptorSetLayoutBinding binding = {};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	binding.descriptorCount = TextureStageCount;
	binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &binding;
	if(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
		LogError << "Could not create Vulkan descriptor set layout";
		return false;
	}

	VkPushConstantRange range = {};
	range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	range.offset = 0;
	range.size = sizeof(VulkanPushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &range;
	if(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
		LogError << "Could not create Vulkan pipeline layout";
		return false;
	}

	VkPipelineCacheCreateInfo cacheInfo = {};
	cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache);

	return true;
}

bool VulkanRenderer::createShaders() {

	auto createModule = [this](const uint32_t * code, size_t size, VkShaderModule & module) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = size;
		createInfo.pCode = code;
		if(vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS) {
			LogError << "Could not create Vulkan shader module";
			module = VK_NULL_HANDLE;
			return false;
		}
		return true;
	};

	return createModule(vk_shader_TexturedVertex_vert, sizeof(vk_shader_TexturedVertex_vert),
	                    m_vertexShaders[VulkanTexturedVertex])
	       && createModule(vk_shader_Vertex_vert, sizeof(vk_shader_Vertex_vert),
	                       m_vertexShaders[VulkanSMYVertex])
	       && createModule(vk_shader_Vertex3_vert, sizeof(vk_shader_Vertex3_vert),
	                       m_vertexShaders[VulkanSMYVertex3])
	       && createModule(vk_shader_Fragment_frag, sizeof(vk_shader_Fragment_frag), m_fragmentShader);
}

void VulkanRenderer::createFrames() {

	for(Frame & f : m_frames) {

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = m_queueFamily;
		vkCreateCommandPool(m_device, &poolInfo, nullptr, &f.commandPool);

		VkCommandBuffer buffers[2];
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = f.commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 2;
		vkAllocateCommandBuffers(m_device, &allocInfo, buffers);
		f.commands = buffers[0];
		f.upload = buffers[1];

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		vkCreateFence(m_device, &fenceInfo, nullptr, &f.fence);

		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &f.imageAvailable);
		vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &f.renderFinished);

	}

}

void VulkanRenderer::createDummyTexture() {

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent = { 1, 1, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &m_dummyImage, &m_dummyAllocation, nullptr);

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_dummyImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	vkCreateImageView(m_device, &viewInfo, nullptr, &m_dummyView);

	VkCommandBuffer cmd = uploadCommands();

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = m_dummyImage;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

	VkClearColorValue white = { { 1.f, 1.f, 1.f, 1.f } };
	vkCmdClearColorImage(cmd, m_dummyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1,
	                     &barrier.subresourceRange);

	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

}

void VulkanRenderer::initialize() {

	std::string apiVersion = vkVersionString(m_properties.apiVersion);

	LogInfo << "Using Vulkan " << apiVersion;
	CrashHandler::setVariable("Vulkan version", apiVersion);

	LogInfo << " ├─ Device: " << m_properties.deviceName;
	CrashHandler::setVariable("Vulkan device", m_properties.deviceName);

	std::string driverVersion = vkVersionString(m_properties.driverVersion);
	if(m_properties.vendorID == 0x10de) {
		// NVIDIA uses a 10.8.8.6 bit encoding
		std::ostringstream oss;
		oss << (m_properties.driverVersion >> 22) << '.' << ((m_properties.driverVersion >> 14) & 0xff);
		driverVersion = oss.str();
	}
	LogInfo << " ├─ Driver: " << driverVersion;

	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memory);
	u64 vram = 0;
	for(uint32_t i = 0; i < memory.memoryHeapCount; i++) {
		if(memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			vram = std::max(vram, u64(memory.memoryHeaps[i].size));
		}
	}
	LogInfo << " └─ VRAM: " << (vram / 1024 / 1024) << " MiB";
	CrashHandler::setVariable("VRAM size", vram);

	credits::setLibraryCredits("graphics", "Vulkan " + apiVersion);

	setMaxAnisotropy(float(config.video.maxAnisotropicFiltering));

}

void VulkanRenderer::beforeResize(bool wasOrIsFullscreen) {
	// The swapchain is recreated lazily, no need to shut down
	ARX_UNUSED(wasOrIsFullscreen);
}

void VulkanRenderer::afterResize() {

	m_swapchainDirty = true;

	if(!isInitialized()) {
		m_TextureStages.resize(TextureStageCount);
		for(size_t i = 0; i < m_TextureStages.size(); ++i) {
			m_TextureStages[i] = std::make_unique<VulkanTextureStage>(unsigned(i));
		}
		onRendererInit();
	}

}

// Swapchain

bool VulkanRenderer::createSwapchain() {

	waitQueueIdle();
	vkDeviceWaitIdle(m_device);

	VkSurfaceCapabilitiesKHR caps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &caps);

	VkExtent2D extent = caps.currentExtent;
	if(extent.width == 0xffffffff) {
		Vec2i size = m_window.getVulkanDrawableSize();
		extent.width = glm::clamp(uint32_t(std::max(int(size.x), 0)), caps.minImageExtent.width, caps.maxImageExtent.width);
		extent.height = glm::clamp(uint32_t(std::max(int(size.y), 0)), caps.minImageExtent.height, caps.maxImageExtent.height);
	}

	if(extent.width == 0 || extent.height == 0) {
		// Minimized
		return false;
	}

	uint32_t presentModeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr);
	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount,
	                                          presentModes.data());
	auto hasPresentMode = [&](VkPresentModeKHR mode) {
		return std::find(presentModes.begin(), presentModes.end(), mode) != presentModes.end();
	};
	m_presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if(m_vsync == 0) {
		if(hasPresentMode(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
			m_presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		} else if(hasPresentMode(VK_PRESENT_MODE_MAILBOX_KHR)) {
			m_presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
		}
	} else if(m_vsync < 0 && hasPresentMode(VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
		m_presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
	}

	static const char * const presentModeNames[] = { "immediate", "mailbox", "fifo", "fifo relaxed" };
	LogInfo << "Vulkan swapchain: " << extent.width << "x" << extent.height << ", vsync " << m_vsync
	        << ", present mode " << (size_t(m_presentMode) < 4 ? presentModeNames[m_presentMode] : "other");

	uint32_t imageCount = caps.minImageCount + 1;
	if(caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
		imageCount = caps.maxImageCount;
	}

	VkSwapchainKHR oldSwapchain = m_swapchain.swapchain;
	destroySwapchain(true);
	
	// Anti-aliasing setting changed: recreate render passes and pipelines
	VkSampleCountFlagBits samples = desiredSampleCount();
	if(samples != m_samples) {
		for(const auto & pipeline : m_pipelines) {
			vkDestroyPipeline(m_device, pipeline.second, nullptr);
		}
		m_pipelines.clear();
		m_boundPipeline = VK_NULL_HANDLE;
		destroyRenderPasses();
		m_samples = samples;
		if(!createRenderPasses()) {
			return false;
		}
	}
	LogInfo << "Vulkan anti-aliasing: " << int(m_samples) << "x";

	m_swapchain.canCopy = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = m_surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = m_swapchain.format;
	createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	if(m_swapchain.canCopy) {
		createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	                          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	if(!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
		for(uint32_t bit = 1; bit != 0; bit <<= 1) {
			if(caps.supportedCompositeAlpha & bit) {
				createInfo.compositeAlpha = VkCompositeAlphaFlagBitsKHR(bit);
				break;
			}
		}
	}
	createInfo.presentMode = m_presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = oldSwapchain;

	VkResult result = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain.swapchain);
	if(oldSwapchain != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(m_device, oldSwapchain, nullptr);
	}
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan swapchain: " << int(result);
		m_swapchain.swapchain = VK_NULL_HANDLE;
		return false;
	}

	m_swapchain.extent = extent;

	uint32_t count = 0;
	vkGetSwapchainImagesKHR(m_device, m_swapchain.swapchain, &count, nullptr);
	m_swapchain.images.resize(count);
	vkGetSwapchainImagesKHR(m_device, m_swapchain.swapchain, &count, m_swapchain.images.data());

	// Depth buffer
	VkImageCreateInfo depthInfo = {};
	depthInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthInfo.imageType = VK_IMAGE_TYPE_2D;
	depthInfo.format = m_depthFormat;
	depthInfo.extent = { extent.width, extent.height, 1 };
	depthInfo.mipLevels = 1;
	depthInfo.arrayLayers = 1;
	depthInfo.samples = m_samples;
	depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo depthAlloc = {};
	depthAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	depthAlloc.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
	if(vmaCreateImage(m_allocator, &depthInfo, &depthAlloc, &m_swapchain.depthImage,
	                  &m_swapchain.depthAllocation, nullptr) != VK_SUCCESS) {
		LogError << "Could not create Vulkan depth buffer";
		return false;
	}

	VkImageViewCreateInfo depthViewInfo = {};
	depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image = m_swapchain.depthImage;
	depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	depthViewInfo.format = m_depthFormat;
	depthViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
	vkCreateImageView(m_device, &depthViewInfo, nullptr, &m_swapchain.depthView);
	
	if(m_samples != VK_SAMPLE_COUNT_1_BIT) {
		VkImageCreateInfo colorInfo = depthInfo;
		colorInfo.format = m_swapchain.format;
		colorInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		if(vmaCreateImage(m_allocator, &colorInfo, &depthAlloc, &m_swapchain.colorImage,
		                  &m_swapchain.colorAllocation, nullptr) != VK_SUCCESS) {
			LogError << "Could not create Vulkan multisample color buffer";
			return false;
		}
		VkImageViewCreateInfo colorViewInfo = depthViewInfo;
		colorViewInfo.image = m_swapchain.colorImage;
		colorViewInfo.format = m_swapchain.format;
		colorViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCreateImageView(m_device, &colorViewInfo, nullptr, &m_swapchain.colorView);
	}

	for(VkImage image : m_swapchain.images) {

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = m_swapchain.format;
		viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		VkImageView view = VK_NULL_HANDLE;
		vkCreateImageView(m_device, &viewInfo, nullptr, &view);
		m_swapchain.views.push_back(view);

		bool msaa = (m_samples != VK_SAMPLE_COUNT_1_BIT);
		VkImageView attachments[] = { msaa ? m_swapchain.colorView : view, m_swapchain.depthView, view };
		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = m_clearRenderPass;
		framebufferInfo.attachmentCount = msaa ? 3 : 2;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = extent.width;
		framebufferInfo.height = extent.height;
		framebufferInfo.layers = 1;
		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &framebuffer);
		m_swapchain.framebuffers.push_back(framebuffer);

	}

	m_swapchainDirty = false;

	return true;
}

void VulkanRenderer::destroySwapchain(bool keepHandle) {

	for(VkFramebuffer framebuffer : m_swapchain.framebuffers) {
		vkDestroyFramebuffer(m_device, framebuffer, nullptr);
	}
	m_swapchain.framebuffers.clear();

	for(VkImageView view : m_swapchain.views) {
		vkDestroyImageView(m_device, view, nullptr);
	}
	m_swapchain.views.clear();
	m_swapchain.images.clear();

	if(m_swapchain.depthView) {
		vkDestroyImageView(m_device, m_swapchain.depthView, nullptr);
		m_swapchain.depthView = VK_NULL_HANDLE;
	}
	if(m_swapchain.depthImage) {
		vmaDestroyImage(m_allocator, m_swapchain.depthImage, m_swapchain.depthAllocation);
		m_swapchain.depthImage = VK_NULL_HANDLE;
		m_swapchain.depthAllocation = VK_NULL_HANDLE;
	}

	if(m_swapchain.colorView) {
		vkDestroyImageView(m_device, m_swapchain.colorView, nullptr);
		m_swapchain.colorView = VK_NULL_HANDLE;
	}
	if(m_swapchain.colorImage) {
		vmaDestroyImage(m_allocator, m_swapchain.colorImage, m_swapchain.colorAllocation);
		m_swapchain.colorImage = VK_NULL_HANDLE;
		m_swapchain.colorAllocation = VK_NULL_HANDLE;
	}
	
	if(keepHandle) {
		m_swapchain.swapchain = VK_NULL_HANDLE;
	} else if(m_swapchain.swapchain) {
		vkDestroySwapchainKHR(m_device, m_swapchain.swapchain, nullptr);
		m_swapchain.swapchain = VK_NULL_HANDLE;
	}

}

void VulkanRenderer::destroy() {

	stopQueueThread();

	if(m_device) {

		vkDeviceWaitIdle(m_device);

		for(VulkanTexture & texture : m_textures) {
			texture.release();
		}
		m_textures.clear();

		m_completedFrame = m_frameNumber;
		processDeferredDestroys();

		destroySwapchain();

		for(Frame & f : m_frames) {
			for(Chunk & chunk : f.chunks) {
				vmaDestroyBuffer(m_allocator, chunk.buffer, chunk.allocation);
			}
			f.chunks.clear();
			for(VkDescriptorPool pool : f.descriptorPools) {
				vkDestroyDescriptorPool(m_device, pool, nullptr);
			}
			f.descriptorPools.clear();
			vkDestroySemaphore(m_device, f.imageAvailable, nullptr);
			vkDestroySemaphore(m_device, f.renderFinished, nullptr);
			vkDestroyFence(m_device, f.fence, nullptr);
			vkDestroyCommandPool(m_device, f.commandPool, nullptr);
		}

		if(m_readbackBuffer) {
			vmaDestroyBuffer(m_allocator, m_readbackBuffer, m_readbackAllocation);
		}

		if(m_dummyView) {
			vkDestroyImageView(m_device, m_dummyView, nullptr);
		}
		if(m_dummyImage) {
			vmaDestroyImage(m_allocator, m_dummyImage, m_dummyAllocation);
		}

		for(const auto & sampler : m_samplers) {
			vkDestroySampler(m_device, sampler.second, nullptr);
		}
		for(const auto & pipeline : m_pipelines) {
			vkDestroyPipeline(m_device, pipeline.second, nullptr);
		}
		for(VkShaderModule module : m_vertexShaders) {
			if(module) {
				vkDestroyShaderModule(m_device, module, nullptr);
			}
		}
		if(m_fragmentShader) {
			vkDestroyShaderModule(m_device, m_fragmentShader, nullptr);
		}
		if(m_pipelineCache) {
			vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
		}
		if(m_pipelineLayout) {
			vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
		}
		if(m_descriptorSetLayout) {
			vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
		}
		destroyRenderPasses();

		if(m_allocator) {
			vmaDestroyAllocator(m_allocator);
		}

		vkDestroyDevice(m_device, nullptr);
		m_device = VK_NULL_HANDLE;
	}

	if(m_instance) {
		if(m_surface) {
			vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
		}
		if(m_debugMessenger) {
			auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
			if(destroyMessenger) {
				destroyMessenger(m_instance, m_debugMessenger, nullptr);
			}
		}
		vkDestroyInstance(m_instance, nullptr);
		m_instance = VK_NULL_HANDLE;
	}

}

// Frame management

void VulkanRenderer::prepareFrame() {

	Frame & f = frame();

	if(f.number != 0) {
		waitSubmitted(f.number);
		vkWaitForFences(m_device, 1, &f.fence, VK_TRUE, UINT64_MAX);
		m_completedFrame = std::max(m_completedFrame, f.number);
	}
	vkResetFences(m_device, 1, &f.fence);

	f.number = m_frameNumber;

	processDeferredDestroys();

	vkResetCommandPool(m_device, f.commandPool, 0);
	f.uploadRecording = false;
	f.commandsRecording = false;

	for(Chunk & chunk : f.chunks) {
		chunk.used = 0;
	}
	f.currentChunk = 0;

	for(VkDescriptorPool pool : f.descriptorPools) {
		vkResetDescriptorPool(m_device, pool, 0);
	}
	f.currentDescriptorPool = 0;
	f.descriptorSets.clear();
	m_textureStateValid = false;

	m_renderPassCount = 0;
	m_skipFrame = false;

}

void VulkanRenderer::processDeferredDestroys() {

	auto it = std::remove_if(m_deferred.begin(), m_deferred.end(), [this](const DeferredDestroy & d) {
		if(d.frame > m_completedFrame) {
			return false;
		}
		if(d.view) {
			vkDestroyImageView(m_device, d.view, nullptr);
		}
		if(d.image) {
			vmaDestroyImage(m_allocator, d.image, d.allocation);
		} else if(d.buffer) {
			vmaDestroyBuffer(m_allocator, d.buffer, d.allocation);
		}
		return true;
	});
	m_deferred.erase(it, m_deferred.end());

}

void VulkanRenderer::deferDestroy(VkBuffer buffer, VmaAllocation allocation) {
	m_deferred.push_back({ m_frameNumber, buffer, VK_NULL_HANDLE, VK_NULL_HANDLE, allocation });
}

void VulkanRenderer::deferDestroy(VkImage image, VkImageView view, VmaAllocation allocation) {
	m_deferred.push_back({ m_frameNumber, VK_NULL_HANDLE, image, view, allocation });
}

VulkanRenderer::StreamAllocation VulkanRenderer::allocateStream(VkDeviceSize size, VkDeviceSize alignment) {

	Frame & f = frame();

	for(; f.currentChunk < f.chunks.size(); f.currentChunk++) {
		Chunk & chunk = f.chunks[f.currentChunk];
		VkDeviceSize offset = (chunk.used + alignment - 1) / alignment * alignment;
		if(offset + size <= chunk.size) {
			chunk.used = offset + size;
			return { chunk.buffer, offset, static_cast<char *>(chunk.data) + offset };
		}
	}

	Chunk chunk;
	chunk.size = std::max(vkMinChunkSize, size);

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = chunk.size;
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
	                   | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo info = {};
	VkResult result = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &chunk.buffer, &chunk.allocation,
	                                  &info);
	if(result != VK_SUCCESS) {
		LogCritical << "Could not allocate Vulkan stream buffer: " << int(result);
		std::abort();
	}
	chunk.data = info.pMappedData;
	chunk.used = size;

	f.chunks.push_back(chunk);
	f.currentChunk = f.chunks.size() - 1;

	return { chunk.buffer, 0, chunk.data };
}

VkCommandBuffer VulkanRenderer::uploadCommands() {

	Frame & f = frame();

	if(!f.uploadRecording) {
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(f.upload, &beginInfo);
		f.uploadRecording = true;
	}

	return f.upload;
}

bool VulkanRenderer::beginRenderPass() {

	if(m_inRenderPass) {
		return true;
	}

	if(m_skipFrame || !m_device) {
		return false;
	}

	Frame & f = frame();

	if(!m_imageAcquired) {

		waitQueueIdle();

		for(int attempt = 0; attempt < 2; attempt++) {

			if(desiredSampleCount() != m_samples) {
				m_swapchainDirty = true;
			}
			if(m_swapchainDirty || m_swapchain.swapchain == VK_NULL_HANDLE) {
				if(!createSwapchain()) {
					m_skipFrame = true;
					return false;
				}
			}

			VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain.swapchain, UINT64_MAX,
			                                        f.imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
			if(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
				if(result == VK_SUBOPTIMAL_KHR) {
					m_swapchainDirty = true;
				}
				m_imageAcquired = true;
				m_acquireWaited = false;
				break;
			}

			if(result == VK_ERROR_OUT_OF_DATE_KHR) {
				m_swapchainDirty = true;
				continue;
			}

			LogError << "Could not acquire Vulkan swapchain image: " << int(result);
			break;
		}

		if(!m_imageAcquired) {
			m_skipFrame = true;
			return false;
		}

	}

	if(!f.commandsRecording) {
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(f.commands, &beginInfo);
		f.commandsRecording = true;
	}

	VkClearValue clearValues[2] = {};
	clearValues[0].color = { { 0.f, 0.f, 0.f, 1.f } };
	clearValues[1].depthStencil = { 1.f, 0 };

	VkRenderPassBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	beginInfo.renderPass = (m_renderPassCount == 0) ? m_clearRenderPass : m_loadRenderPass;
	beginInfo.framebuffer = m_swapchain.framebuffers[m_imageIndex];
	beginInfo.renderArea = { { 0, 0 }, m_swapchain.extent };
	beginInfo.clearValueCount = 2;
	beginInfo.pClearValues = clearValues;
	vkCmdBeginRenderPass(f.commands, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);

	m_inRenderPass = true;
	m_renderPassCount++;

	m_boundPipeline = VK_NULL_HANDLE;
	m_boundDescriptorSet = VK_NULL_HANDLE;
	m_boundVertexBuffer = VK_NULL_HANDLE;
	m_pushConstantsValid = false;
	m_dynamicStateValid = false;

	return true;
}

void VulkanRenderer::endRenderPass() {
	if(m_inRenderPass) {
		vkCmdEndRenderPass(frame().commands);
		m_inRenderPass = false;
	}
}

void VulkanRenderer::executeQueueJob(const QueueJob & job) {

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = (job.waitSemaphore != VK_NULL_HANDLE) ? 1 : 0;
	submitInfo.pWaitSemaphores = &job.waitSemaphore;
	submitInfo.pWaitDstStageMask = &waitStage;
	submitInfo.commandBufferCount = job.bufferCount;
	submitInfo.pCommandBuffers = job.buffers;
	submitInfo.signalSemaphoreCount = job.present ? 1 : 0;
	submitInfo.pSignalSemaphores = &job.signalSemaphore;

	VkResult result = vkQueueSubmit(m_queue, 1, &submitInfo, job.fence);
	if(result != VK_SUCCESS) {
		LogCritical << "Vulkan queue submit failed: " << int(result);
		std::abort();
	}

	if(job.present) {
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &job.signalSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &job.swapchain;
		presentInfo.pImageIndices = &job.imageIndex;
		result = vkQueuePresentKHR(m_queue, &presentInfo);
		if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			m_presentOutOfDate = true;
		} else if(result != VK_SUCCESS) {
			LogError << "Vulkan present failed: " << int(result);
		}
	}

}

void VulkanRenderer::queueThread() {

	std::unique_lock<std::mutex> lock(m_queueMutex);

	while(true) {

		m_queueCondition.wait(lock, [this] { return m_queueStop || !m_queueJobs.empty(); });
		if(m_queueJobs.empty()) {
			break; // Stop requested and all work done
		}

		QueueJob job = m_queueJobs.front();
		m_queueBusy = true;
		lock.unlock();

		executeQueueJob(job);

		lock.lock();
		m_queueJobs.pop_front();
		m_queueBusy = false;
		m_submittedFrame = std::max(m_submittedFrame, job.frame);
		m_queueCondition.notify_all();

	}

}

void VulkanRenderer::waitQueueIdle() {

	if(m_queueThread.joinable()) {
		std::unique_lock<std::mutex> lock(m_queueMutex);
		m_queueCondition.wait(lock, [this] { return m_queueJobs.empty() && !m_queueBusy; });
	}

	if(m_presentOutOfDate.exchange(false)) {
		m_swapchainDirty = true;
	}

}

void VulkanRenderer::waitSubmitted(uint64_t frame) {
	if(m_queueThread.joinable()) {
		std::unique_lock<std::mutex> lock(m_queueMutex);
		m_queueCondition.wait(lock, [this, frame] { return m_submittedFrame >= frame; });
	}
}

void VulkanRenderer::stopQueueThread() {
	if(m_queueThread.joinable()) {
		{
			std::lock_guard<std::mutex> lock(m_queueMutex);
			m_queueStop = true;
		}
		m_queueCondition.notify_all();
		m_queueThread.join();
	}
}

void VulkanRenderer::submit(bool present, bool wait) {

	Frame & f = frame();

	endRenderPass();

	QueueJob job;
	job.frame = m_frameNumber;
	job.bufferCount = 0;
	if(f.uploadRecording) {
		vkEndCommandBuffer(f.upload);
		job.buffers[job.bufferCount++] = f.upload;
		f.uploadRecording = false;
	}
	if(f.commandsRecording) {
		vkEndCommandBuffer(f.commands);
		job.buffers[job.bufferCount++] = f.commands;
		f.commandsRecording = false;
	}

	bool waitAcquire = m_imageAcquired && !m_acquireWaited;
	job.present = present && m_imageAcquired;
	job.waitSemaphore = waitAcquire ? f.imageAvailable : VK_NULL_HANDLE;
	job.signalSemaphore = f.renderFinished;
	job.fence = f.fence;
	job.swapchain = m_swapchain.swapchain;
	job.imageIndex = m_imageIndex;

	if(waitAcquire) {
		m_acquireWaited = true;
	}
	if(job.present) {
		m_imageAcquired = false;
	}

	if(wait || !m_queueThread.joinable()) {
		// Synchronous submission: let the queue thread finish first
		waitQueueIdle();
		executeQueueJob(job);
		if(m_presentOutOfDate.exchange(false)) {
			m_swapchainDirty = true;
		}
	} else {
		// Submit and present on the queue thread so that the next frame can start immediately
		{
			std::lock_guard<std::mutex> lock(m_queueMutex);
			m_queueJobs.push_back(job);
		}
		m_queueCondition.notify_all();
	}

	if(wait) {
		vkWaitForFences(m_device, 1, &f.fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &f.fence);
		m_completedFrame = std::max(m_completedFrame, m_frameNumber - 1);
	}

}

void VulkanRenderer::present() {

	if(!m_device) {
		return;
	}

	submit(true, false);

	m_frameNumber++;
	prepareFrame();

}

void VulkanRenderer::setVSync(int vsync) {
	if(vsync != m_vsync) {
		m_vsync = vsync;
		m_swapchainDirty = true;
	}
}

// State

void VulkanRenderer::SetViewMatrix(const glm::mat4x4 & matView) {
	m_view = matView;
	m_viewProjection = m_projection * m_view;
}

void VulkanRenderer::SetProjectionMatrix(const glm::mat4x4 & matProj) {
	m_projection = matProj;
	m_viewProjection = m_projection * m_view;
}

void VulkanRenderer::ReleaseAllTextures() {
	for(VulkanTexture & texture : m_textures) {
		texture.destroy();
	}
}

void VulkanRenderer::RestoreAllTextures() {
	for(VulkanTexture & texture : m_textures) {
		texture.restore();
	}
}

void VulkanRenderer::reloadColorKeyTextures() {
	for(VulkanTexture & texture : m_textures) {
		if(texture.hasColorKey()) {
			texture.restore();
		}
	}
}

Texture * VulkanRenderer::createTexture() {
	VulkanTexture * texture = new VulkanTexture(this);
	m_textures.push_back(*texture);
	return texture;
}

void VulkanRenderer::SetViewport(const Rect & viewport) {
	if(viewport != m_viewport) {
		m_viewport = viewport;
		m_dynamicStateValid = false;
	}
}

void VulkanRenderer::SetScissor(const Rect & rect) {
	if(rect != m_scissor) {
		m_scissor = rect;
		m_dynamicStateValid = false;
	}
}

void VulkanRenderer::Clear(BufferFlags bufferFlags, Color clearColor, float clearDepth, size_t nrects,
                           Rect * rect) {

	if(!beginRenderPass()) {
		return;
	}

	VkClearAttachment attachments[2];
	uint32_t attachmentCount = 0;

	if(bufferFlags & ColorBuffer) {
		Color4f color(clearColor);
		VkClearAttachment & attachment = attachments[attachmentCount++];
		attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		attachment.colorAttachment = 0;
		attachment.clearValue.color = { { color.r, color.g, color.b, color.a } };
	}

	if(bufferFlags & DepthBuffer) {
		VkClearAttachment & attachment = attachments[attachmentCount++];
		attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		attachment.colorAttachment = 0;
		attachment.clearValue.depthStencil = { clearDepth, 0 };
	}

	if(attachmentCount == 0) {
		return;
	}

	const int32_t width = int32_t(m_swapchain.extent.width);
	const int32_t height = int32_t(m_swapchain.extent.height);

	std::vector<VkClearRect> rects;
	if(nrects == 0) {
		rects.push_back({ { { 0, 0 }, m_swapchain.extent }, 0, 1 });
	} else {
		for(size_t i = 0; i < nrects; i++) {
			int32_t left = glm::clamp(int32_t(rect[i].left), int32_t(0), width);
			int32_t top = glm::clamp(int32_t(rect[i].top), int32_t(0), height);
			int32_t right = glm::clamp(int32_t(rect[i].right), int32_t(0), width);
			int32_t bottom = glm::clamp(int32_t(rect[i].bottom), int32_t(0), height);
			if(right > left && bottom > top) {
				rects.push_back({ { { left, top }, { uint32_t(right - left), uint32_t(bottom - top) } }, 0, 1 });
			}
		}
	}

	if(!rects.empty()) {
		vkCmdClearAttachments(frame().commands, attachmentCount, attachments, uint32_t(rects.size()), rects.data());
	}

}

void VulkanRenderer::SetFogColor(Color color) {
	m_fogColor = Color4f(color);
}

void VulkanRenderer::SetFogParams(float fogStart, float fogEnd) {
	m_fogStart = fogStart;
	m_fogEnd = fogEnd;
}

void VulkanRenderer::SetAntialiasing(bool enable) {
	// The sample count is fixed per render pass - this only selects how alpha cutouts are handled,
	// like toggling GL_MULTISAMPLE in the OpenGL renderer
	m_antialiasingEnabled = enable;
}

VkSampleCountFlagBits VulkanRenderer::desiredSampleCount() const {
	return config.video.antialiasing ? m_maxSamples : VK_SAMPLE_COUNT_1_BIT;
}

VulkanRenderer::AlphaTestMode VulkanRenderer::alphaTestMode() const {
	
	if(!m_state.getAlphaCutout()) {
		return AlphaTestNone;
	}
	
	bool msaa = m_antialiasingEnabled && m_samples != VK_SAMPLE_COUNT_1_BIT && config.video.antialiasing;
	bool useSS = msaa && m_hasSampleShading && config.video.alphaCutoutAntialiasing == int(CrispAlphaCutoutAA);
	bool useA2C = msaa && config.video.alphaCutoutAntialiasing == int(FuzzyAlphaCutoutAA);
	
	// When blending, transparent texels must still be discarded but alpha to coverage cannot
	// be used as the source alpha could be applied twice.
	BlendingFactor blendSrc = m_state.getBlendSrc();
	BlendingFactor blendDst = m_state.getBlendDst();
	if(blendSrc == BlendOne && blendDst == BlendZero) {
		return useSS ? AlphaTestSampleShading : useA2C ? AlphaTestCoverage : AlphaTestStrict;
	} else if(blendSrc == BlendOne && blendDst == BlendOne) {
		return AlphaTestConservative;
	} else if(blendSrc == BlendSrcAlpha || blendDst == BlendInvSrcAlpha) {
		return AlphaTestConservative;
	}
	return useSS ? AlphaTestSampleShading : AlphaTestStrict;
}

void VulkanRenderer::SetFillMode(FillMode mode) {
	m_fillMode = mode;
}

void VulkanRenderer::setMaxAnisotropy(float value) {
	m_maxAnisotropy = glm::clamp(value, 1.f, m_maxSupportedAnisotropy);
	m_textureStateValid = false;
}

Renderer::AlphaCutoutAntialising VulkanRenderer::getMaxSupportedAlphaCutoutAntialiasing() const {
	return m_hasSampleShading ? CrispAlphaCutoutAA : FuzzyAlphaCutoutAA;
}

bool VulkanRenderer::hasFormatSupport(VkFormat format, VkFormatFeatureFlags features) const {
	VkFormatProperties properties;
	vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &properties);
	return (properties.optimalTilingFeatures & features) == features;
}

// Vertex buffers

std::unique_ptr<VertexBuffer<TexturedVertex>> VulkanRenderer::createVertexBufferTL(size_t capacity,
                                                                                   BufferUsage usage) {
	return std::make_unique<VulkanVertexBuffer<TexturedVertex>>(this, capacity, usage);
}

std::unique_ptr<VertexBuffer<SMY_VERTEX>> VulkanRenderer::createVertexBuffer(size_t capacity,
                                                                             BufferUsage usage) {
	return std::make_unique<VulkanVertexBuffer<SMY_VERTEX>>(this, capacity, usage);
}

std::unique_ptr<VertexBuffer<SMY_VERTEX3>> VulkanRenderer::createVertexBuffer3(size_t capacity,
                                                                               BufferUsage usage) {
	return std::make_unique<VulkanVertexBuffer<SMY_VERTEX3>>(this, capacity, usage);
}

void VulkanRenderer::drawIndexed(Primitive primitive, const TexturedVertex * vertices, size_t nvertices,
                                 unsigned short * indices, size_t nindices) {

	if(nvertices == 0 || nindices == 0 || !beginRenderPass()) {
		return;
	}

	VkDeviceSize size = nvertices * sizeof(TexturedVertex);
	StreamAllocation allocation = allocateStream(size, sizeof(TexturedVertex));
	std::memcpy(allocation.data, vertices, size_t(size));

	size_t firstVertex = size_t(allocation.offset / sizeof(TexturedVertex));
	drawVertices(VulkanTexturedVertex, primitive, allocation.buffer, firstVertex, nvertices, indices, nindices);

}

// Pipelines and draw calls

VkPipeline VulkanRenderer::getPipeline(VulkanVertexType type, VkPrimitiveTopology topology,
                                       AlphaTestMode alphaTest) {

	RenderState state = m_state;

	BlendingFactor blendSrc = state.getBlendSrc();
	BlendingFactor blendDst = state.getBlendDst();
	if(state.getAlphaCutout() && blendSrc == BlendOne && blendDst == BlendOne) {
		blendSrc = BlendSrcAlpha;
	}

	bool wireframe = (m_fillMode == FillWireframe) && m_hasFillModeNonSolid;
	bool isLine = (topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);

	u64 key = u64(type)
	        | (u64(topology) << 2)
	        | (u64(state.getCull() && !isLine) << 6)
	        | (u64(state.getDepthTest()) << 7)
	        | (u64(state.getDepthWrite()) << 8)
	        | (u64(blendSrc) << 9)
	        | (u64(blendDst) << 13)
	        | (u64(wireframe) << 17)
	        | (u64(alphaTest) << 18);

	auto it = m_pipelines.find(key);
	if(it != m_pipelines.end()) {
		return it->second;
	}

	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = m_vertexShaders[type];
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = m_fragmentShader;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding = {};
	binding.binding = 0;
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	std::vector<VkVertexInputAttributeDescription> attributes;
	switch(type) {
		case VulkanTexturedVertex: {
			binding.stride = sizeof(TexturedVertex);
			attributes.push_back({ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, uint32_t(offsetof(TexturedVertex, p)) });
			attributes.push_back({ 1, 0, VK_FORMAT_R8G8B8A8_UNORM, uint32_t(offsetof(TexturedVertex, color)) });
			attributes.push_back({ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(TexturedVertex, uv)) });
			break;
		}
		case VulkanSMYVertex: {
			binding.stride = sizeof(SMY_VERTEX);
			attributes.push_back({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(SMY_VERTEX, p)) });
			attributes.push_back({ 1, 0, VK_FORMAT_R8G8B8A8_UNORM, uint32_t(offsetof(SMY_VERTEX, color)) });
			attributes.push_back({ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(SMY_VERTEX, uv)) });
			break;
		}
		case VulkanSMYVertex3: {
			binding.stride = sizeof(SMY_VERTEX3);
			attributes.push_back({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(SMY_VERTEX3, p)) });
			attributes.push_back({ 1, 0, VK_FORMAT_R8G8B8A8_UNORM, uint32_t(offsetof(SMY_VERTEX3, color)) });
			attributes.push_back({ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(SMY_VERTEX3, uv)) });
			attributes.push_back({ 3, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(SMY_VERTEX3, uv) + sizeof(Vec2f)) });
			attributes.push_back({ 4, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(SMY_VERTEX3, uv) + 2 * sizeof(Vec2f)) });
			break;
		}
		default: arx_unreachable();
	}

	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = uint32_t(attributes.size());
	vertexInput.pVertexAttributeDescriptions = attributes.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = topology;

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
	// Same convention as OpenGL thanks to the flipped viewport
	rasterizer.cullMode = (state.getCull() && !isLine) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_TRUE;
	rasterizer.lineWidth = 1.f;

	VkPipelineMultisampleStateCreateInfo multisample = {};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = m_samples;
	multisample.alphaToCoverageEnable = (alphaTest == AlphaTestCoverage) ? VK_TRUE : VK_FALSE;
	multisample.sampleShadingEnable = (alphaTest == AlphaTestSampleShading) ? VK_TRUE : VK_FALSE;
	multisample.minSampleShading = 1.f;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	// Like the OpenGL renderer, keep the depth test enabled so depth writes work without it
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = state.getDepthWrite() ? VK_TRUE : VK_FALSE;
	depthStencil.depthCompareOp = state.getDepthTest() ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_ALWAYS;

	VkPipelineColorBlendAttachmentState blend = {};
	blend.blendEnable = (blendSrc != BlendOne || blendDst != BlendZero) ? VK_TRUE : VK_FALSE;
	blend.srcColorBlendFactor = arxToVkBlendFactor[blendSrc];
	blend.dstColorBlendFactor = arxToVkBlendFactor[blendDst];
	blend.colorBlendOp = VK_BLEND_OP_ADD;
	// VK_BLEND_FACTOR_SRC_ALPHA_SATURATE is treated as one for alpha by OpenGL
	blend.srcAlphaBlendFactor = (blendSrc == BlendSrcAlphaSaturate) ? VK_BLEND_FACTOR_ONE
	                                                                 : arxToVkBlendFactor[blendSrc];
	blend.dstAlphaBlendFactor = arxToVkBlendFactor[blendDst];
	blend.alphaBlendOp = VK_BLEND_OP_ADD;
	blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
	                       | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlend = {};
	colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlend.attachmentCount = 1;
	colorBlend.pAttachments = &blend;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
	                                   VK_DYNAMIC_STATE_DEPTH_BIAS };
	VkPipelineDynamicStateCreateInfo dynamicState = {};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 3;
	dynamicState.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	createInfo.stageCount = 2;
	createInfo.pStages = stages;
	createInfo.pVertexInputState = &vertexInput;
	createInfo.pInputAssemblyState = &inputAssembly;
	createInfo.pViewportState = &viewportState;
	createInfo.pRasterizationState = &rasterizer;
	createInfo.pMultisampleState = &multisample;
	createInfo.pDepthStencilState = &depthStencil;
	createInfo.pColorBlendState = &colorBlend;
	createInfo.pDynamicState = &dynamicState;
	createInfo.layout = m_pipelineLayout;
	createInfo.renderPass = m_clearRenderPass;
	createInfo.subpass = 0;

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult result = vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &createInfo, nullptr, &pipeline);
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan pipeline: " << int(result);
		pipeline = VK_NULL_HANDLE;
	}

	m_pipelines[key] = pipeline;

	return pipeline;
}

VkSampler VulkanRenderer::getSampler(const VulkanTextureStage * stage, const VulkanTexture * texture) {

	TextureStage::WrapMode wrap = TextureStage::WrapRepeat;
	TextureStage::FilterMode minFilter = TextureStage::FilterLinear;
	TextureStage::FilterMode magFilter = TextureStage::FilterLinear;
	bool mipmaps = false;
	float lodBias = 0.f;
	float anisotropy = 1.f;

	if(stage && texture) {
		// Match the OpenGL renderer: non-power-of-two textures are always clamped
		bool isPOT = vkIsPowerOfTwo(texture->getSize().x) && vkIsPowerOfTwo(texture->getSize().y);
		wrap = isPOT ? stage->getWrapMode() : TextureStage::WrapClamp;
		minFilter = stage->getMinFilter();
		magFilter = stage->getMagFilter();
		mipmaps = texture->mipLevels() > 1;
		lodBias = stage->lodBias();
		if(mipmaps) {
			anisotropy = m_maxAnisotropy;
		}
	}

	int32_t biasKey = int32_t(lodBias * 256.f);
	uint32_t anisotropyKey = uint32_t(anisotropy * 4.f);
	u64 key = u64(wrap)
	        | (u64(minFilter) << 2)
	        | (u64(magFilter) << 3)
	        | (u64(mipmaps) << 4)
	        | (u64(anisotropyKey & 0xffff) << 8)
	        | (u64(uint32_t(biasKey)) << 24);

	auto it = m_samplers.find(key);
	if(it != m_samplers.end()) {
		return it->second;
	}

	VkSamplerCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	createInfo.magFilter = arxToVkFilter[magFilter];
	createInfo.minFilter = arxToVkFilter[minFilter];
	// Like OpenGL: GL_NEAREST_MIPMAP_LINEAR / GL_LINEAR_MIPMAP_LINEAR
	createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	createInfo.addressModeU = arxToVkWrapMode[wrap];
	createInfo.addressModeV = arxToVkWrapMode[wrap];
	createInfo.addressModeW = arxToVkWrapMode[wrap];
	createInfo.mipLodBias = glm::clamp(lodBias, -m_properties.limits.maxSamplerLodBias,
	                                   m_properties.limits.maxSamplerLodBias);
	createInfo.anisotropyEnable = (anisotropy > 1.f) ? VK_TRUE : VK_FALSE;
	createInfo.maxAnisotropy = anisotropy;
	createInfo.minLod = 0.f;
	createInfo.maxLod = mipmaps ? VK_LOD_CLAMP_NONE : 0.f;
	createInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

	VkSampler sampler = VK_NULL_HANDLE;
	if(vkCreateSampler(m_device, &createInfo, nullptr, &sampler) != VK_SUCCESS) {
		LogError << "Could not create Vulkan sampler";
	}

	m_samplers[key] = sampler;

	return sampler;
}

VkDescriptorSet VulkanRenderer::getDescriptorSet(const DescriptorKey & key) {

	Frame & f = frame();

	auto it = f.descriptorSets.find(key);
	if(it != f.descriptorSets.end()) {
		return it->second;
	}

	VkDescriptorSet set = VK_NULL_HANDLE;

	for(; ; f.currentDescriptorPool++) {

		if(f.currentDescriptorPool == f.descriptorPools.size()) {
			VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			                                  vkDescriptorPoolSize * TextureStageCount };
			VkDescriptorPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.maxSets = vkDescriptorPoolSize;
			poolInfo.poolSizeCount = 1;
			poolInfo.pPoolSizes = &poolSize;
			VkDescriptorPool pool = VK_NULL_HANDLE;
			if(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
				LogCritical << "Could not create Vulkan descriptor pool";
				std::abort();
			}
			f.descriptorPools.push_back(pool);
		}

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = f.descriptorPools[f.currentDescriptorPool];
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &m_descriptorSetLayout;
		if(vkAllocateDescriptorSets(m_device, &allocInfo, &set) == VK_SUCCESS) {
			break;
		}

	}

	VkDescriptorImageInfo images[TextureStageCount];
	for(size_t i = 0; i < TextureStageCount; i++) {
		images[i].sampler = key.samplers[i];
		images[i].imageView = key.views[i];
		images[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	VkWriteDescriptorSet write = {};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = set;
	write.dstBinding = 0;
	write.descriptorCount = TextureStageCount;
	write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write.pImageInfo = images;
	vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

	f.descriptorSets[key] = set;

	return set;
}

void VulkanRenderer::flushState(VulkanVertexType type, VkPrimitiveTopology topology) {

	VkCommandBuffer cmd = frame().commands;

	AlphaTestMode alphaTest = alphaTestMode();
	
	// Pipeline - only look it up when its inputs changed
	if(m_boundPipeline == VK_NULL_HANDLE || m_state != m_pipelineState || type != m_pipelineType
	   || topology != m_pipelineTopology || m_fillMode != m_pipelineFillMode
	   || alphaTest != m_pipelineAlphaTest) {
		VkPipeline pipeline = getPipeline(type, topology, alphaTest);
		if(pipeline != m_boundPipeline) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			m_boundPipeline = pipeline;
		}
		m_pipelineState = m_state;
		m_pipelineType = type;
		m_pipelineTopology = topology;
		m_pipelineFillMode = m_fillMode;
		m_pipelineAlphaTest = alphaTest;
	}

	// Dynamic state
	if(!m_dynamicStateValid) {

		VkViewport viewport;
		Rect rect = m_viewport;
		if(rect.width() <= 0 || rect.height() <= 0) {
			rect = Rect(int32_t(m_swapchain.extent.width), int32_t(m_swapchain.extent.height));
		}
		// Flip the y axis to match OpenGL conventions
		viewport.x = float(rect.left);
		viewport.y = float(rect.bottom);
		viewport.width = float(rect.width());
		viewport.height = -float(rect.height());
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		// Screen-space vertices: same transform as the OpenGL renderer
		glm::mat4 transform(1.f);
		transform = glm::translate(transform, glm::vec3(-1.f, 1.f, 0.f));
		transform = glm::scale(transform, glm::vec3(2.f / float(rect.width()), -2.f / float(rect.height()), 1.f));
		m_screenTransform = glm::translate(transform, glm::vec3(0.5f, 0.5f, 0.f));

		VkRect2D scissor = { { 0, 0 }, m_swapchain.extent };
		if(m_scissor.isValid()) {
			int32_t width = int32_t(m_swapchain.extent.width);
			int32_t height = int32_t(m_swapchain.extent.height);
			int32_t left = glm::clamp(int32_t(m_scissor.left), int32_t(0), width);
			int32_t top = glm::clamp(int32_t(m_scissor.top), int32_t(0), height);
			int32_t right = glm::clamp(int32_t(m_scissor.right), int32_t(left), width);
			int32_t bottom = glm::clamp(int32_t(m_scissor.bottom), int32_t(top), height);
			scissor = { { left, top }, { uint32_t(right - left), uint32_t(bottom - top) } };
		}
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		vkCmdSetDepthBias(cmd, -float(m_state.getDepthOffset()), 0.f, -float(m_state.getDepthOffset()));
		m_boundDepthOffset = m_state.getDepthOffset();

		m_dynamicStateValid = true;

	} else if(m_boundDepthOffset != m_state.getDepthOffset()) {
		vkCmdSetDepthBias(cmd, -float(m_state.getDepthOffset()), 0.f, -float(m_state.getDepthOffset()));
		m_boundDepthOffset = m_state.getDepthOffset();
	}

	// Textures - only resolve samplers and descriptor sets when the texture state changed
	TextureState textureState;
	for(size_t i = 0; i < TextureStageCount; i++) {
		VulkanTextureStage * stage = GetTextureStage(i);
		StageState & s = textureState.stages[i];
		if(stage) {
			s.texture = stage->texture();
			s.view = s.texture ? s.texture->view() : VK_NULL_HANDLE;
			s.lodBias = stage->lodBias();
			s.wrap = uint8_t(stage->getWrapMode());
			s.minFilter = uint8_t(stage->getMinFilter());
			s.magFilter = uint8_t(stage->getMagFilter());
			s.colorOp = uint8_t(stage->colorOp());
			s.alphaOp = uint8_t(stage->alphaOp());
		}
	}

	if(!m_textureStateValid || std::memcmp(&textureState, &m_textureState, sizeof(textureState)) != 0) {
		DescriptorKey key;
		uint32_t ops = 0;
		for(size_t i = 0; i < TextureStageCount; i++) {
			VulkanTextureStage * stage = GetTextureStage(i);
			VulkanTexture * texture = textureState.stages[i].view ? textureState.stages[i].texture : nullptr;
			key.views[i] = texture ? texture->view() : m_dummyView;
			key.samplers[i] = getSampler(texture ? stage : nullptr, texture);
			if(texture && stage->isEnabled()) {
				ops |= (uint32_t(stage->colorOp()) | (uint32_t(stage->alphaOp()) << 4)) << (8 * i);
			}
		}
		m_textureDescriptorSet = getDescriptorSet(key);
		m_textureOps = ops;
		m_textureState = textureState;
		m_textureStateValid = true;
	}

	if(m_textureDescriptorSet != m_boundDescriptorSet) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
		                        &m_textureDescriptorSet, 0, nullptr);
		m_boundDescriptorSet = m_textureDescriptorSet;
	}

	// Push constants
	VulkanPushConstants pc;
	if(type == VulkanTexturedVertex) {
		pc.transform = m_screenTransform;
		pc.viewZ = glm::vec4(0.f);
	} else {
		pc.transform = m_viewProjection;
		pc.viewZ = glm::vec4(m_view[0][2], m_view[1][2], m_view[2][2], m_view[3][2]);
	}
	pc.fogColor = glm::vec4(m_fogColor.r, m_fogColor.g, m_fogColor.b, m_fogColor.a);

	float alphaRef = -1.f;
	switch(alphaTest) {
		case AlphaTestNone: alphaRef = -1.f; break;
		case AlphaTestConservative:
		case AlphaTestCoverage: alphaRef = 0.f; break;
		case AlphaTestStrict:
		case AlphaTestSampleShading: alphaRef = 0.5f; break;
	}
	pc.params = glm::vec4(m_fogStart, m_fogEnd, alphaRef, m_state.getFog() ? 1.f : 0.f);
	pc.ops[0] = m_textureOps;
	pc.ops[1] = pc.ops[2] = pc.ops[3] = 0;

	if(!m_pushConstantsValid || std::memcmp(&pc, &m_pushConstants, sizeof(pc)) != 0) {
		vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(pc), &pc);
		m_pushConstants = pc;
		m_pushConstantsValid = true;
	}

}

void VulkanRenderer::drawVertices(VulkanVertexType type, Primitive primitive, VkBuffer buffer,
                                  size_t firstVertex, size_t vertexCount,
                                  const unsigned short * indices, size_t indexCount) {

	if(vertexCount == 0 || (indices && indexCount == 0)) {
		return;
	}

	if(!beginRenderPass()) {
		return;
	}

	VkPrimitiveTopology topology;
	switch(primitive) {
		case TriangleList:  topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
		case TriangleStrip: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
		case TriangleFan:   topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
		case LineList:      topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
		case LineStrip:     topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
		default: arx_unreachable();
	}

	if(primitive == TriangleFan) {
		// Triangle fans are not supported everywhere (e.g. MoltenVK) - convert to lists
		size_t count = indices ? indexCount : vertexCount;
		if(count < 3) {
			return;
		}
		m_indexScratch.clear();
		for(size_t i = 1; i + 1 < count; i++) {
			m_indexScratch.push_back(indices ? indices[0] : 0);
			m_indexScratch.push_back(indices ? indices[i] : (unsigned short)i);
			m_indexScratch.push_back(indices ? indices[i + 1] : (unsigned short)(i + 1));
		}
		indices = m_indexScratch.data();
		indexCount = m_indexScratch.size();
	}

	flushState(type, topology);

	if(m_boundPipeline == VK_NULL_HANDLE) {
		return;
	}

	VkCommandBuffer cmd = frame().commands;

	if(buffer != m_boundVertexBuffer) {
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
		m_boundVertexBuffer = buffer;
	}

	if(indices) {
		VkDeviceSize size = indexCount * sizeof(unsigned short);
		StreamAllocation allocation = allocateStream(size, 4);
		std::memcpy(allocation.data, indices, size_t(size));
		vkCmdBindIndexBuffer(cmd, allocation.buffer, allocation.offset, VK_INDEX_TYPE_UINT16);
		vkCmdDrawIndexed(cmd, uint32_t(indexCount), 1, 0, int32_t(firstVertex), 0);
	} else {
		vkCmdDraw(cmd, uint32_t(vertexCount), 1, uint32_t(firstVertex), 0);
	}

}

// Screenshots

bool VulkanRenderer::getSnapshot(Image & image) {

	if(!beginRenderPass() || !m_swapchain.canCopy) {
		return false;
	}

	endRenderPass();

	const uint32_t width = m_swapchain.extent.width;
	const uint32_t height = m_swapchain.extent.height;
	VkDeviceSize size = VkDeviceSize(width) * height * 4;

	if(size > m_readbackSize) {
		if(m_readbackBuffer) {
			vmaDestroyBuffer(m_allocator, m_readbackBuffer, m_readbackAllocation);
			m_readbackBuffer = VK_NULL_HANDLE;
		}
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		if(vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &m_readbackBuffer, &m_readbackAllocation,
		                   nullptr) != VK_SUCCESS) {
			m_readbackBuffer = VK_NULL_HANDLE;
			m_readbackSize = 0;
			return false;
		}
		m_readbackSize = size;
	}

	VkCommandBuffer cmd = frame().commands;
	VkImage swapchainImage = m_swapchain.images[m_imageIndex];

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = swapchainImage;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

	VkBufferImageCopy region = {};
	region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.imageExtent = { width, height, 1 };
	vkCmdCopyImageToBuffer(cmd, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_readbackBuffer, 1, &region);

	barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

	VkBufferMemoryBarrier bufferBarrier = {};
	bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarrier.buffer = m_readbackBuffer;
	bufferBarrier.size = VK_WHOLE_SIZE;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
	                     0, nullptr, 1, &bufferBarrier, 0, nullptr);

	submit(false, true);

	vmaInvalidateAllocation(m_allocator, m_readbackAllocation, 0, VK_WHOLE_SIZE);
	VmaAllocationInfo info;
	vmaGetAllocationInfo(m_allocator, m_readbackAllocation, &info);
	const u8 * src = static_cast<const u8 *>(info.pMappedData);

	image.create(size_t(width), size_t(height), Image::Format_R8G8B8);
	u8 * dst = image.getData();
	bool bgr = (m_swapchain.format == VK_FORMAT_B8G8R8A8_UNORM || m_swapchain.format == VK_FORMAT_B8G8R8A8_SRGB);
	for(size_t i = 0, n = size_t(width) * height; i < n; i++, src += 4, dst += 3) {
		dst[0] = bgr ? src[2] : src[0];
		dst[1] = src[1];
		dst[2] = bgr ? src[0] : src[2];
	}

	return true;
}

bool VulkanRenderer::getSnapshot(Image & image, size_t width, size_t height) {

	Image fullsize;
	if(!getSnapshot(fullsize)) {
		return false;
	}

	image.resizeFrom(fullsize, width, height);

	return true;
}
