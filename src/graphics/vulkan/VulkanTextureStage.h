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

#ifndef ARX_GRAPHICS_VULKAN_VULKANTEXTURESTAGE_H
#define ARX_GRAPHICS_VULKAN_VULKANTEXTURESTAGE_H

#include "graphics/texture/TextureStage.h"

class VulkanTexture;

/*!
 * Texture stage state - applied by the fragment shader and sampler selection
 */
class VulkanTextureStage final : public TextureStage {

public:

	explicit VulkanTextureStage(unsigned int stage);

	Texture * getTexture() const override;
	void setTexture(Texture * texture) override;
	void resetTexture() override;

	void setColorOp(TextureOp textureOp) override { m_colorOp = textureOp; }
	void setAlphaOp(TextureOp textureOp) override { m_alphaOp = textureOp; }

	void setMipMapLODBias(float bias) override { m_lodBias = bias; }

	VulkanTexture * texture() const { return m_texture; }
	TextureOp colorOp() const { return m_colorOp; }
	TextureOp alphaOp() const { return m_alphaOp; }
	float lodBias() const { return m_lodBias; }

	bool isEnabled() const { return m_colorOp != OpDisable || m_alphaOp != OpDisable; }

private:

	VulkanTexture * m_texture;
	TextureOp m_colorOp;
	TextureOp m_alphaOp;
	float m_lodBias;

	friend class VulkanTexture;

};

#endif // ARX_GRAPHICS_VULKAN_VULKANTEXTURESTAGE_H
