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

#include "graphics/vulkan/VulkanTexture.h"

#include <algorithm>
#include <cstring>

#include "graphics/vulkan/VulkanRenderer.h"
#include "graphics/vulkan/VulkanTextureStage.h"
#include "io/log/Logger.h"

VulkanTexture::VulkanTexture(VulkanRenderer * renderer)
	: m_renderer(renderer)
	, m_vkImage(VK_NULL_HANDLE)
	, m_view(VK_NULL_HANDLE)
	, m_allocation(VK_NULL_HANDLE)
	, m_mipLevels(1)
{ }

VulkanTexture::~VulkanTexture() {
	destroy();
}

bool VulkanTexture::create() {
	// Vulkan supports non-power-of-two textures, no need to pad
	m_storedSize = getSize();
	return getSize().x > 0 && getSize().y > 0;
}

namespace {

struct VulkanFormatInfo {
	VkFormat format;
	VkComponentMapping swizzle;
	size_t bpp;
};

constexpr VkComponentSwizzle vkSwizzleR = VK_COMPONENT_SWIZZLE_R;
constexpr VkComponentSwizzle vkSwizzleG = VK_COMPONENT_SWIZZLE_G;
constexpr VkComponentSwizzle vkSwizzleZero = VK_COMPONENT_SWIZZLE_ZERO;
constexpr VkComponentSwizzle vkSwizzleOne = VK_COMPONENT_SWIZZLE_ONE;
constexpr VkComponentMapping vkSwizzleIdentity = {
	VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
	VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
};

} // anonymous namespace

void VulkanTexture::upload() {

	// Choose a Vulkan format matching the OpenGL fixed-function semantics

	// Three-channel formats are expanded to four channels when copying to staging memory
	// as they are not widely supported for sampling.
	bool expandRGB = (getFormat() == Image::Format_R8G8B8 || getFormat() == Image::Format_B8G8R8);

	VulkanFormatInfo info;
	if(isIntensity()) {
		info = { VK_FORMAT_R8_UNORM, { vkSwizzleR, vkSwizzleR, vkSwizzleR, vkSwizzleR }, 1 };
	} else if(getFormat() == Image::Format_L8) {
		info = { VK_FORMAT_R8_UNORM, { vkSwizzleR, vkSwizzleR, vkSwizzleR, vkSwizzleOne }, 1 };
	} else if(getFormat() == Image::Format_A8) {
		info = { VK_FORMAT_R8_UNORM, { vkSwizzleZero, vkSwizzleZero, vkSwizzleZero, vkSwizzleR }, 1 };
	} else if(getFormat() == Image::Format_L8A8) {
		info = { VK_FORMAT_R8G8_UNORM, { vkSwizzleR, vkSwizzleR, vkSwizzleR, vkSwizzleG }, 2 };
	} else if(getFormat() == Image::Format_R8G8B8 || getFormat() == Image::Format_R8G8B8A8) {
		info = { VK_FORMAT_R8G8B8A8_UNORM, vkSwizzleIdentity, 4 };
	} else if(getFormat() == Image::Format_B8G8R8 || getFormat() == Image::Format_B8G8R8A8) {
		info = { VK_FORMAT_B8G8R8A8_UNORM, vkSwizzleIdentity, 4 };
	} else {
		arx_assert_msg(false, "Unsupported image format: %ld", long(getFormat()));
		return;
	}

	if(!m_image.isValid() || m_image.getWidth() != size_t(getSize().x)
	   || m_image.getHeight() != size_t(getSize().y)) {
		LogError << "Invalid texture image";
		return;
	}

	// Low-end GPUs: store color textures as 16-bit to halve memory use and bandwidth.
	// Textures with smooth alpha keep 8 bits per channel as 4-bit alpha visibly bands.
	enum PackMode { PackNone, Pack565, Pack5551 };
	PackMode pack = PackNone;
	const bool rgbOrder = (getFormat() == Image::Format_R8G8B8 || getFormat() == Image::Format_R8G8B8A8);
	const bool isColor = rgbOrder || getFormat() == Image::Format_B8G8R8 || getFormat() == Image::Format_B8G8R8A8;
	if(isColor && !isIntensity() && m_renderer->useCompactTextures()) {
		const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
		                                    | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
		                                    | VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
		if(expandRGB) {
			if(m_renderer->hasFormatSupport(VK_FORMAT_R5G6B5_UNORM_PACK16, needed)) {
				pack = Pack565;
				info = { VK_FORMAT_R5G6B5_UNORM_PACK16, vkSwizzleIdentity, 2 };
			}
		} else if(m_renderer->hasFormatSupport(VK_FORMAT_A1R5G5B5_UNORM_PACK16, needed)) {
			const unsigned char * src = m_image.getData();
			bool binaryAlpha = true;
			for(size_t i = 0, n = m_image.getWidth() * m_image.getHeight(); i < n; i++) {
				if(src[i * 4 + 3] != 0 && src[i * 4 + 3] != 0xff) {
					binaryAlpha = false;
					break;
				}
			}
			if(binaryAlpha) {
				pack = Pack5551;
				info = { VK_FORMAT_A1R5G5B5_UNORM_PACK16, vkSwizzleIdentity, 2 };
			}
		}
	}

	// Each upload uses a new image: the old one may still be in use by the GPU
	release();

	uint32_t width = uint32_t(getSize().x);
	uint32_t height = uint32_t(getSize().y);

	m_mipLevels = 1;
	if(hasMipmaps()) {
		uint32_t size = std::max(width, height);
		while(size > 1) {
			size /= 2;
			m_mipLevels++;
		}
	}

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = info.format;
	imageInfo.extent = { width, height, 1 };
	imageInfo.mipLevels = m_mipLevels;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	if(m_mipLevels > 1) {
		imageInfo.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	VkResult result = vmaCreateImage(m_renderer->allocator(), &imageInfo, &allocInfo,
	                                 &m_vkImage, &m_allocation, nullptr);
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan texture " << getFileName() << ": " << int(result);
		m_vkImage = VK_NULL_HANDLE;
		m_allocation = VK_NULL_HANDLE;
		return;
	}

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_vkImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = info.format;
	viewInfo.components = info.swizzle;
	viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_mipLevels, 0, 1 };
	result = vkCreateImageView(m_renderer->device(), &viewInfo, nullptr, &m_view);
	if(result != VK_SUCCESS) {
		LogError << "Could not create Vulkan texture view: " << int(result);
		m_view = VK_NULL_HANDLE;
		release();
		return;
	}

	// Copy the pixels to staging memory

	VkDeviceSize size = VkDeviceSize(width) * height * info.bpp;
	VulkanRenderer::StreamAllocation staging = m_renderer->allocateStream(size, 16);
	if(pack != PackNone) {
		const unsigned char * src = m_image.getData();
		const size_t srcBpp = (pack == Pack565) ? 3 : 4;
		const size_t ri = rgbOrder ? 0 : 2;
		const size_t bi = rgbOrder ? 2 : 0;
		uint16_t * dst = static_cast<uint16_t *>(staging.data);
		for(size_t i = 0, n = size_t(width) * height; i < n; i++, src += srcBpp) {
			uint16_t r = uint16_t(src[ri] >> 3);
			uint16_t b = uint16_t(src[bi] >> 3);
			if(pack == Pack565) {
				dst[i] = uint16_t((r << 11) | (uint16_t(src[1] >> 2) << 5) | b);
			} else {
				uint16_t a = (src[3] != 0) ? 1 : 0;
				dst[i] = uint16_t((a << 15) | (r << 10) | (uint16_t(src[1] >> 3) << 5) | b);
			}
		}
	} else if(expandRGB) {
		const unsigned char * src = m_image.getData();
		unsigned char * dst = static_cast<unsigned char *>(staging.data);
		for(size_t i = 0, n = size_t(width) * height; i < n; i++, src += 3, dst += 4) {
			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
			dst[3] = 0xff;
		}
	} else {
		arx_assert(m_image.getSize() == size_t(size));
		std::memcpy(staging.data, m_image.getData(), size_t(size));
	}

	VkCommandBuffer cmd = m_renderer->uploadCommands();

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = m_vkImage;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_mipLevels, 0, 1 };
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

	VkBufferImageCopy region = {};
	region.bufferOffset = staging.offset;
	region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.imageExtent = { width, height, 1 };
	vkCmdCopyBufferToImage(cmd, staging.buffer, m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Generate mipmaps by repeatedly blitting the previous level
	int32_t mipWidth = int32_t(width);
	int32_t mipHeight = int32_t(height);
	for(uint32_t level = 1; level < m_mipLevels; level++) {

		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1 };
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
		                     0, nullptr, 0, nullptr, 1, &barrier);

		int32_t nextWidth = std::max(mipWidth / 2, 1);
		int32_t nextHeight = std::max(mipHeight / 2, 1);

		VkImageBlit blit = {};
		blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 };
		blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
		blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
		blit.dstOffsets[1] = { nextWidth, nextHeight, 1 };
		vkCmdBlitImage(cmd, m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		                     0, nullptr, 0, nullptr, 1, &barrier);

		mipWidth = nextWidth;
		mipHeight = nextHeight;
	}

	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, m_mipLevels - 1, 1, 0, 1 };
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
	                     0, nullptr, 0, nullptr, 1, &barrier);

}

void VulkanTexture::release() {
	if(m_vkImage != VK_NULL_HANDLE || m_view != VK_NULL_HANDLE) {
		m_renderer->deferDestroy(m_vkImage, m_view, m_allocation);
		m_vkImage = VK_NULL_HANDLE;
		m_view = VK_NULL_HANDLE;
		m_allocation = VK_NULL_HANDLE;
	}
}

void VulkanTexture::destroy() {

	release();

	for(size_t i = 0; i < m_renderer->getTextureStageCount(); i++) {
		VulkanTextureStage * stage = m_renderer->GetTextureStage(i);
		if(stage->m_texture == this) {
			stage->m_texture = nullptr;
		}
	}

}
