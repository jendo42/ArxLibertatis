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

#ifndef ARX_GRAPHICS_VULKAN_VULKANRENDERER_H
#define ARX_GRAPHICS_VULKAN_VULKANRENDERER_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/intrusive/list.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

#include "graphics/Renderer.h"
#include "graphics/vulkan/VulkanTexture.h"
#include "math/Rectangle.h"
#include "math/Types.h"
#include "platform/Platform.h"

class VulkanTextureStage;

/*!
 * Interface the window system has to provide for the Vulkan renderer.
 */
class VulkanWindow {

public:

	virtual ~VulkanWindow() = default;

	//! Append the instance extensions needed to create a surface for the window
	virtual bool getVulkanInstanceExtensions(std::vector<const char *> & extensions) = 0;

	//! Create a surface for the window or return VK_NULL_HANDLE
	virtual VkSurfaceKHR createVulkanSurface(VkInstance instance) = 0;

	//! Size of the drawable area in pixels
	virtual Vec2i getVulkanDrawableSize() = 0;

};

//! Must match the push constant block in shaders/Common.glsl
struct VulkanPushConstants {
	glm::mat4 transform;
	glm::vec4 viewZ;
	glm::vec4 fogColor;
	glm::vec4 params; // fog start, fog end, alpha reference, fog enabled
	uint32_t ops[4];
};
static_assert(sizeof(VulkanPushConstants) == 128, "push constants must fit in the guaranteed 128 bytes");

enum VulkanVertexType {
	VulkanTexturedVertex,
	VulkanSMYVertex,
	VulkanSMYVertex3,
	VulkanVertexTypeCount
};

template <class Vertex>
struct VulkanVertexTraits;

template <>
struct VulkanVertexTraits<TexturedVertex> { static constexpr VulkanVertexType type = VulkanTexturedVertex; };

template <>
struct VulkanVertexTraits<SMY_VERTEX> { static constexpr VulkanVertexType type = VulkanSMYVertex; };

template <>
struct VulkanVertexTraits<SMY_VERTEX3> { static constexpr VulkanVertexType type = VulkanSMYVertex3; };

class VulkanRenderer final : public Renderer {

public:

	static constexpr size_t FramesInFlight = 2;
	static constexpr size_t TextureStageCount = 3;

	explicit VulkanRenderer(VulkanWindow & window);
	~VulkanRenderer() override;

	/*!
	 * Create the Vulkan instance, surface and device.
	 * Must be called after the window has been created and before \ref initialize().
	 * \return false if Vulkan is not available.
	 */
	bool create();

	void initialize() override;

	void beforeResize(bool wasOrIsFullscreen) override;
	void afterResize() override;

	// Matrices
	void SetViewMatrix(const glm::mat4x4 & matView) override;
	void SetProjectionMatrix(const glm::mat4x4 & matProj) override;

	// Texture management
	void ReleaseAllTextures() override;
	void RestoreAllTextures() override;
	void reloadColorKeyTextures() override;

	// Factory
	Texture * createTexture() override;

	// Viewport
	void SetViewport(const Rect & viewport) override;

	void SetScissor(const Rect & rect) override;

	// Render Target
	void Clear(BufferFlags bufferFlags, Color clearColor = Color(), float clearDepth = 1.f,
	           size_t nrects = 0, Rect * rect = 0) override;

	// Fog
	void SetFogColor(Color color) override;
	void SetFogParams(float fogStart, float fogEnd) override;

	// Rasterizer
	void SetAntialiasing(bool enable) override;
	void SetFillMode(FillMode mode) override;

	float getMaxAnisotropy() const { return m_maxAnisotropy; }
	float getMaxSupportedAnisotropy() const override { return m_maxSupportedAnisotropy; }
	void setMaxAnisotropy(float value) override;

	AlphaCutoutAntialising getMaxSupportedAlphaCutoutAntialiasing() const override;

	std::unique_ptr<VertexBuffer<TexturedVertex>> createVertexBufferTL(size_t capacity, BufferUsage usage) override;
	std::unique_ptr<VertexBuffer<SMY_VERTEX>> createVertexBuffer(size_t capacity, BufferUsage usage) override;
	std::unique_ptr<VertexBuffer<SMY_VERTEX3>> createVertexBuffer3(size_t capacity, BufferUsage usage) override;

	void drawIndexed(Primitive primitive, const TexturedVertex * vertices, size_t nvertices,
	                 unsigned short * indices, size_t nindices) override;

	bool getSnapshot(Image & image) override;
	bool getSnapshot(Image & image, size_t width, size_t height) override;

	//! Submit the current frame and present it
	void present();

	//! 0: no vsync, 1: vsync, -1: adaptive vsync
	void setVSync(int vsync);

	// Internal interface for textures and vertex buffers

	VulkanTextureStage * GetTextureStage(size_t textureStage) {
		return reinterpret_cast<VulkanTextureStage *>(Renderer::GetTextureStage(textureStage));
	}

	VkDevice device() const { return m_device; }
	VmaAllocator allocator() const { return m_allocator; }

	//! Number of the frame currently being recorded
	u64 currentFrame() const { return m_frameNumber; }

	//! Check if the GPU has finished all work for a frame
	bool isFrameComplete(u64 frame) const { return frame <= m_completedFrame; }

	void deferDestroy(VkBuffer buffer, VmaAllocation allocation);
	void deferDestroy(VkImage image, VkImageView view, VmaAllocation allocation);

	struct StreamAllocation {
		VkBuffer buffer;
		VkDeviceSize offset;
		void * data;
	};

	//! Allocate host-visible memory that is valid until the end of the current frame
	StreamAllocation allocateStream(VkDeviceSize size, VkDeviceSize alignment = 16);

	//! Command buffer for resource uploads, submitted before the current frame
	VkCommandBuffer uploadCommands();

	/*!
	 * Draw vertices from a buffer
	 *
	 * \param firstVertex index of the first vertex in the buffer
	 * \param indices optional 16-bit indices relative to firstVertex
	 */
	void drawVertices(VulkanVertexType type, Primitive primitive, VkBuffer buffer,
	                  size_t firstVertex, size_t vertexCount,
	                  const unsigned short * indices = nullptr, size_t indexCount = 0);

	bool hasFormatSupport(VkFormat format, VkFormatFeatureFlags features) const;

private:

	struct Chunk {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		void * data = nullptr;
		VkDeviceSize size = 0;
		VkDeviceSize used = 0;
	};

	struct DeferredDestroy {
		u64 frame;
		VkBuffer buffer;
		VkImage image;
		VkImageView view;
		VmaAllocation allocation;
	};

	struct DescriptorKey {
		VkImageView views[TextureStageCount];
		VkSampler samplers[TextureStageCount];
		bool operator==(const DescriptorKey & o) const;
	};

	struct DescriptorKeyHash {
		size_t operator()(const DescriptorKey & key) const;
	};

	struct Frame {
		VkCommandPool commandPool = VK_NULL_HANDLE;
		VkCommandBuffer commands = VK_NULL_HANDLE;
		VkCommandBuffer upload = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;
		VkSemaphore imageAvailable = VK_NULL_HANDLE;
		VkSemaphore renderFinished = VK_NULL_HANDLE;
		u64 number = 0;
		bool uploadRecording = false;
		bool commandsRecording = false;
		std::vector<Chunk> chunks;
		size_t currentChunk = 0;
		std::vector<VkDescriptorPool> descriptorPools;
		size_t currentDescriptorPool = 0;
		std::unordered_map<DescriptorKey, VkDescriptorSet, DescriptorKeyHash> descriptorSets;
	};

	struct Swapchain {
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		VkExtent2D extent = { 0, 0 };
		bool canCopy = false;
		std::vector<VkImage> images;
		std::vector<VkImageView> views;
		std::vector<VkFramebuffer> framebuffers;
		VkImage depthImage = VK_NULL_HANDLE;
		VkImageView depthView = VK_NULL_HANDLE;
		VmaAllocation depthAllocation = VK_NULL_HANDLE;
		// Multisampled color buffer, resolved to the swapchain image
		VkImage colorImage = VK_NULL_HANDLE;
		VkImageView colorView = VK_NULL_HANDLE;
		VmaAllocation colorAllocation = VK_NULL_HANDLE;
	};

	bool createInstance();
	bool selectPhysicalDevice();
	bool createDevice();
	bool createRenderPasses();
	void destroyRenderPasses();
	//! Sample count requested by the configuration
	VkSampleCountFlagBits desiredSampleCount() const;
	
	enum AlphaTestMode {
		AlphaTestNone,
		AlphaTestConservative, //!< Discard fully transparent fragments
		AlphaTestStrict,       //!< Discard fragments with alpha <= 0.5
		AlphaTestCoverage,     //!< Alpha to coverage
		AlphaTestSampleShading //!< Per-sample alpha test
	};
	//! Same alpha cutout logic as the OpenGL renderer
	AlphaTestMode alphaTestMode() const;
	bool createPipelineLayout();
	bool createShaders();
	void createFrames();
	void createDummyTexture();

	bool createSwapchain();
	void destroySwapchain(bool keepHandle = false);

	void destroy();

	Frame & frame() { return m_frames[m_frameNumber % FramesInFlight]; }

	void prepareFrame();
	bool beginRenderPass();
	void endRenderPass();
	void submit(bool present, bool wait);
	
	struct QueueJob {
		uint64_t frame;
		VkCommandBuffer buffers[2];
		uint32_t bufferCount;
		VkSemaphore waitSemaphore;
		VkSemaphore signalSemaphore;
		VkFence fence;
		bool present;
		VkSwapchainKHR swapchain;
		uint32_t imageIndex;
	};
	
	void executeQueueJob(const QueueJob & job);
	void queueThread();
	//! Wait until the queue thread has no pending work (required before using the queue or swapchain)
	void waitQueueIdle();
	//! Wait until the work for a frame has been submitted
	void waitSubmitted(uint64_t frame);
	void stopQueueThread();
	void processDeferredDestroys();

	VkPipeline getPipeline(VulkanVertexType type, VkPrimitiveTopology topology, AlphaTestMode alphaTest);
	VkSampler getSampler(const VulkanTextureStage * stage, const VulkanTexture * texture);
	VkDescriptorSet getDescriptorSet(const DescriptorKey & key);
	void flushState(VulkanVertexType type, VkPrimitiveTopology topology);

	VulkanWindow & m_window;

	VkInstance m_instance;
	VkDebugUtilsMessengerEXT m_debugMessenger;
	VkSurfaceKHR m_surface;
	VkPhysicalDevice m_physicalDevice;
	VkPhysicalDeviceProperties m_properties;
	VkDevice m_device;
	uint32_t m_queueFamily;
	VkQueue m_queue;
	VmaAllocator m_allocator;

	bool m_hasFillModeNonSolid;
	bool m_hasAnisotropy;
	bool m_hasDebugUtils;
	bool m_hasPortabilitySubset;
	bool m_hasSampleShading;
	VkSampleCountFlagBits m_maxSamples;
	VkSampleCountFlagBits m_samples; //!< Sample count of the current render passes
	bool m_antialiasingEnabled; //!< Set by SetAntialiasing()
	VkFormat m_depthFormat;

	VkRenderPass m_clearRenderPass;
	VkRenderPass m_loadRenderPass;
	VkDescriptorSetLayout m_descriptorSetLayout;
	VkPipelineLayout m_pipelineLayout;
	VkPipelineCache m_pipelineCache;
	VkShaderModule m_vertexShaders[VulkanVertexTypeCount];
	VkShaderModule m_fragmentShader;
	std::unordered_map<u64, VkPipeline> m_pipelines;
	std::unordered_map<u64, VkSampler> m_samplers;

	// 1x1 white texture bound to unused texture stages
	VkImage m_dummyImage;
	VkImageView m_dummyView;
	VmaAllocation m_dummyAllocation;

	Swapchain m_swapchain;
	bool m_swapchainDirty;
	VkPresentModeKHR m_presentMode;
	int m_vsync;

	std::array<Frame, FramesInFlight> m_frames;
	u64 m_frameNumber;
	u64 m_completedFrame;
	std::vector<DeferredDestroy> m_deferred;

	uint32_t m_imageIndex;
	bool m_imageAcquired;
	bool m_acquireWaited;
	bool m_inRenderPass;
	bool m_skipFrame;
	uint32_t m_renderPassCount;

	// Queue submission and presentation run on a separate thread
	std::thread m_queueThread;
	std::mutex m_queueMutex;
	std::condition_variable m_queueCondition;
	std::deque<QueueJob> m_queueJobs;
	bool m_queueStop;
	bool m_queueBusy;
	uint64_t m_submittedFrame;
	std::atomic<bool> m_presentOutOfDate;
	
	VkBuffer m_readbackBuffer;
	VmaAllocation m_readbackAllocation;
	VkDeviceSize m_readbackSize;

	// Current state
	Rect m_viewport;
	Rect m_scissor;
	glm::mat4x4 m_projection;
	glm::mat4x4 m_view;
	glm::mat4x4 m_viewProjection;
	glm::mat4x4 m_screenTransform;
	Color4f m_fogColor;
	float m_fogStart;
	float m_fogEnd;
	FillMode m_fillMode;
	float m_maxAnisotropy;
	float m_maxSupportedAnisotropy;

	// State bound in the current command buffer
	VkPipeline m_boundPipeline;
	VkDescriptorSet m_boundDescriptorSet;
	VkBuffer m_boundVertexBuffer;
	VulkanPushConstants m_pushConstants;
	bool m_pushConstantsValid;
	bool m_dynamicStateValid;
	unsigned m_boundDepthOffset;

	// Inputs of the bound pipeline
	RenderState m_pipelineState;
	VulkanVertexType m_pipelineType;
	VkPrimitiveTopology m_pipelineTopology;
	FillMode m_pipelineFillMode;
	AlphaTestMode m_pipelineAlphaTest;

	// Texture stage state of the last draw, compared with memcmp (all padding is explicit)
	struct StageState {
		VulkanTexture * texture = nullptr;
		VkImageView view = VK_NULL_HANDLE;
		float lodBias = 0.f;
		uint8_t wrap = 0;
		uint8_t minFilter = 0;
		uint8_t magFilter = 0;
		uint8_t colorOp = 0;
		uint8_t alphaOp = 0;
		uint8_t padding[7] = {};
	};
	static_assert(sizeof(StageState) == 32, "StageState must not have implicit padding");
	struct TextureState {
		StageState stages[TextureStageCount];
	};
	TextureState m_textureState;
	bool m_textureStateValid;
	VkDescriptorSet m_textureDescriptorSet;
	uint32_t m_textureOps;

	std::vector<unsigned short> m_indexScratch;

	typedef boost::intrusive::list<VulkanTexture, boost::intrusive::constant_time_size<false>> TextureList;
	TextureList m_textures;

	friend class VulkanTexture;

};

#endif // ARX_GRAPHICS_VULKAN_VULKANRENDERER_H
