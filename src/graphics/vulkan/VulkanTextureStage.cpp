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

#include "graphics/vulkan/VulkanTextureStage.h"

#include "graphics/vulkan/VulkanTexture.h"

VulkanTextureStage::VulkanTextureStage(unsigned int stage)
	: TextureStage(stage)
	, m_texture(nullptr)
	// Match the default state of the OpenGL renderer
	, m_colorOp(stage == 0 ? OpModulate : OpDisable)
	, m_alphaOp(stage == 0 ? OpSelectArg1 : OpDisable)
	, m_lodBias(0.f)
{ }

Texture * VulkanTextureStage::getTexture() const {
	return m_texture;
}

void VulkanTextureStage::setTexture(Texture * texture) {
	arx_assert(texture != nullptr);
	m_texture = static_cast<VulkanTexture *>(texture);
}

void VulkanTextureStage::resetTexture() {
	m_texture = nullptr;
}
