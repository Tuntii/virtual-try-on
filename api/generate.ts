/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import type { VercelRequest, VercelResponse } from '@vercel/node';

export const config = {
  maxDuration: 60,
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { modelImage, modelMime, clothingImage, clothingMime } = req.body as {
    modelImage: string;
    modelMime: string;
    clothingImage: string;
    clothingMime: string;
  };

  if (!modelImage || !clothingImage) {
    return res.status(400).json({ error: 'modelImage ve clothingImage gereklidir.' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Sunucu yapılandırma hatası.' });
  }

  const ai = new GoogleGenAI({ apiKey });

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.0-flash-preview-image-generation',
      contents: {
        parts: [
          {
            inlineData: {
              data: modelImage,
              mimeType: modelMime || 'image/jpeg',
            },
          },
          {
            inlineData: {
              data: clothingImage,
              mimeType: clothingMime || 'image/jpeg',
            },
          },
          {
            text: 'A high-end professional studio fashion photography shot. The person from the first image is wearing the exact clothing item from the second image. Flawless lighting, clean studio background, highly detailed, photorealistic.',
          },
        ],
      },
      config: {
        imageConfig: {
          aspectRatio: '3:4',
          imageSize: '1K',
        },
      },
    });

    const parts = response.candidates?.[0]?.content?.parts ?? [];
    for (const part of parts) {
      if (part.inlineData?.data) {
        return res.status(200).json({ image: part.inlineData.data });
      }
    }

    return res.status(500).json({ error: 'Yapay zeka bir görsel döndürmedi.' });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Görsel oluşturma sırasında hata oluştu.';
    return res.status(500).json({ error: message });
  }
}
