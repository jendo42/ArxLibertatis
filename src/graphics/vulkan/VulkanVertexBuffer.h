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

#ifndef ARX_GRAPHICS_VULKAN_VULKANVERTEXBUFFER_H
#define ARX_GRAPHICS_VULKAN_VULKANVERTEXBUFFER_H

#include <cstring>
#include <vector>

#include "graphics/Vertex.h"
#include "graphics/VertexBuffer.h"
#include "graphics/vulkan/VulkanRenderer.h"
#include "io/log/Logger.h"

/*!
 * Host-visible vertex buffer.
 *
 * Emulates the OpenGL buffer semantics: when the GPU may still be reading the
 * current storage, DiscardBuffer switches to a different storage ("orphaning")
 * and writes without any flags switch to a copy of the current storage.
 * NoOverwrite writes go directly to the current storage.
 */
template <class Vertex>
class VulkanVertexBuffer final : public VertexBuffer<Vertex> {

	typedef VertexBuffer<Vertex> Base;

	struct Storage {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		Vertex * data = nullptr;
		u64 lastUsed = 0;
	};

public:

	using Base::capacity;

	VulkanVertexBuffer(VulkanRenderer * renderer, size_t capacity, Renderer::BufferUsage usage)
		: Base(capacity)
		, m_renderer(renderer)
		, m_usage(usage)
		, m_current(0)
	{
		m_storage.push_back(allocate());
	}

	~VulkanVertexBuffer() override {
		for(Storage & storage : m_storage) {
			if(storage.buffer != VK_NULL_HANDLE) {
				m_renderer->deferDestroy(storage.buffer, storage.allocation);
			}
		}
	}

	void setData(const Vertex * vertices, size_t count, size_t offset, BufferFlags flags) override {
		arx_assert(offset < capacity());
		arx_assert(offset + count <= capacity());
		prepareWrite(flags);
		if(count > 0) {
			std::memcpy(m_storage[m_current].data + offset, vertices, count * sizeof(Vertex));
		}
	}

	Vertex * lock(BufferFlags flags, size_t offset, size_t count) override {
		ARX_UNUSED(count);
		arx_assert(offset < capacity());
		prepareWrite(flags);
		return m_storage[m_current].data + offset;
	}

	void unlock() override { }

	void draw(Renderer::Primitive primitive, size_t count, size_t offset) const override {
		arx_assert(offset + count <= capacity());
		Storage & storage = m_storage[m_current];
		storage.lastUsed = m_renderer->currentFrame();
		m_renderer->drawVertices(VulkanVertexTraits<Vertex>::type, primitive, storage.buffer, offset, count);
	}

	void drawIndexed(Renderer::Primitive primitive, size_t count, size_t offset,
	                 const unsigned short * indices, size_t nbindices) const override {
		arx_assert(offset + count <= capacity());
		arx_assert(indices != nullptr);
		Storage & storage = m_storage[m_current];
		storage.lastUsed = m_renderer->currentFrame();
		m_renderer->drawVertices(VulkanVertexTraits<Vertex>::type, primitive, storage.buffer, offset, count,
		                         indices, nbindices);
	}

private:

	Storage allocate() {

		Storage storage;

		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = std::max<VkDeviceSize>(capacity() * sizeof(Vertex), 16);
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		VmaAllocationInfo info = {};
		VkResult result = vmaCreateBuffer(m_renderer->allocator(), &bufferInfo, &allocInfo,
		                                  &storage.buffer, &storage.allocation, &info);
		if(result != VK_SUCCESS) {
			LogCritical << "Could not allocate Vulkan vertex buffer: " << int(result);
			std::abort();
		}

		storage.data = reinterpret_cast<Vertex *>(info.pMappedData);

		return storage;
	}

	bool isBusy(const Storage & storage) const {
		return storage.lastUsed != 0 && !m_renderer->isFrameComplete(storage.lastUsed);
	}

	void prepareWrite(BufferFlags flags) {

		if(!isBusy(m_storage[m_current]) || (flags & NoOverwrite)) {
			return;
		}

		size_t next = m_storage.size();
		for(size_t i = 0; i < m_storage.size(); i++) {
			if(i != m_current && !isBusy(m_storage[i])) {
				next = i;
				break;
			}
		}
		if(next == m_storage.size()) {
			m_storage.push_back(allocate());
		}

		if(!(flags & DiscardBuffer)) {
			// Preserve the existing contents
			std::memcpy(m_storage[next].data, m_storage[m_current].data, capacity() * sizeof(Vertex));
		}

		m_current = next;
	}

	VulkanRenderer * m_renderer;
	Renderer::BufferUsage m_usage;
	mutable std::vector<Storage> m_storage;
	size_t m_current;

};

#endif // ARX_GRAPHICS_VULKAN_VULKANVERTEXBUFFER_H
