/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import PreviewArea from './components/PreviewArea';

const MAX_VARIATIONS = 6;

export default function App() {
  const [modelImage, setModelImage] = useState<string | null>(null);
  const [modelMime, setModelMime] = useState<string | null>(null);

  const [clothingImage, setClothingImage] = useState<string | null>(null);
  const [clothingMime, setClothingMime] = useState<string | null>(null);

  const [variations, setVariations] = useState<string[]>([]);
  const [activeVariationIndex, setActiveVariationIndex] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (isVariation = false) => {
    if (!modelImage || !clothingImage) return;
    if (isVariation && variations.length >= MAX_VARIATIONS) return;

    setIsGenerating(true);
    setError(null);
    if (!isVariation) {
      setVariations([]);
      setActiveVariationIndex(0);
    }

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelImage,
          modelMime: modelMime ?? 'image/jpeg',
          clothingImage,
          clothingMime: clothingMime ?? 'image/jpeg',
        }),
      });

      const data = await response.json() as { image?: string; error?: string };

      if (!response.ok) {
        throw new Error(data.error ?? 'Görsel oluşturma sırasında hata oluştu.');
      }

      if (!data.image) {
        throw new Error('Yapay zeka bir görsel döndürmedi.');
      }

      const newImage = `data:image/png;base64,${data.image}`;
      setVariations(prev => {
        const next = isVariation ? [...prev, newImage] : [newImage];
        setActiveVariationIndex(next.length - 1);
        return next;
      });

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Görsel oluşturma sırasında hata oluştu.';
      setError(message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen lg:h-screen bg-[#080808] text-[#E0E0E0] font-sans flex flex-col overflow-hidden select-none selection:bg-white/30 selection:text-white">
      {/* Navigation Bar */}
      <nav className="h-16 border-b border-white/10 flex items-center justify-between px-6 lg:px-8 bg-[#0A0A0A] shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-white flex items-center justify-center rounded-sm">
            <span className="text-black font-bold text-lg leading-none">V</span>
          </div>
          <span className="text-xl tracking-[0.2em] font-serif italic text-white/90">STÜDYO</span>
        </div>
        <div className="hidden sm:flex items-center space-x-6 text-[11px] uppercase tracking-widest text-white/50 font-medium">
          <span className="text-white hover:text-white transition-colors cursor-pointer">Sanal Deneme</span>
          <span className="hover:text-white cursor-pointer transition-colors">Koleksiyonlar</span>
          <div className="h-4 w-[1px] bg-white/20" />
          <span className="text-white/80">Gemini AI</span>
        </div>
      </nav>

      {/* Main Workspace */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0 overflow-y-auto lg:overflow-hidden h-full">
        {/* Sidebar Controls */}
        <aside className="lg:col-span-4 xl:col-span-3 border-r border-white/10 bg-[#0A0A0A] p-6 lg:p-8 flex flex-col h-full lg:overflow-y-auto w-full">
          <div className="space-y-8">
            <ImageUpload
              step="01"
              label="Model / Manken"
              image={modelImage}
              mime={modelMime}
              placeholder="Manken veya Kendi Fotoğrafın"
              hint="Önerilen: Stüdyo ışığında çekilmiş net görseller."
              onFile={(b64, m) => { setModelImage(b64); setModelMime(m); }}
              onClear={() => { setModelImage(null); setModelMime(null); }}
            />
            <ImageUpload
              step="02"
              label="Ürün Görseli"
              image={clothingImage}
              mime={clothingMime}
              placeholder="T-shirt veya Üst Giyim"
              hint="Önerilen: Temiz arka plana sahip kıyafet."
              onFile={(b64, m) => { setClothingImage(b64); setClothingMime(m); }}
              onClear={() => { setClothingImage(null); setClothingMime(null); }}
            />
          </div>

          <div className="mt-8 lg:mt-auto pt-8 space-y-4">
            <button
              type="button"
              onClick={() => handleGenerate(false)}
              disabled={!modelImage || !clothingImage || isGenerating}
              className="w-full py-4 bg-white text-black text-[11px] uppercase tracking-[0.2em] font-bold hover:bg-white/90 transition-all disabled:opacity-30 disabled:hover:bg-white disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>İŞLENİYOR...</span>
                </>
              ) : (
                <span>GİYDİR VE OLUŞTUR</span>
              )}
            </button>
            {error && (
              <div className="p-3 border border-red-500/30 text-red-400 text-xs text-center rounded-sm">
                {error}
              </div>
            )}
          </div>
        </aside>

        <PreviewArea
          variations={variations}
          activeIndex={activeVariationIndex}
          isGenerating={isGenerating}
          onVariationSelect={setActiveVariationIndex}
          onAddVariation={() => handleGenerate(true)}
        />
      </main>

      {/* Footer Bar */}
      <footer className="hidden lg:flex h-10 border-t border-white/10 bg-[#0A0A0A] items-center justify-between px-8 shrink-0">
        <div className="flex items-center space-x-4 text-[9px] text-white/30 uppercase tracking-widest">
          <span>Lat: 41.0082° N</span>
          <span>Lon: 28.9784° E</span>
          <span>Cloud TPU v4 Active</span>
        </div>
        <div className="text-[9px] text-white/30 uppercase tracking-widest italic font-serif">
          V-Stüdyo Digital Atelier © 2024
        </div>
      </footer>
    </div>
  );
}
