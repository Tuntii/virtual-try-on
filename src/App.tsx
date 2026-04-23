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

        {/* Preview Area */}
        <section className="lg:col-span-8 xl:col-span-9 bg-[#050505] relative flex flex-col items-center justify-center p-6 lg:p-12 min-h-[60vh] lg:min-h-0 lg:h-full lg:overflow-hidden">
          {/* Top Info */}
          <div className="hidden lg:flex absolute top-8 left-8 space-x-12 z-10 p-4 lg:p-0 pointer-events-none">
            <div>
              <p className="text-[10px] text-white/30 uppercase tracking-widest">Çözünürlük</p>
              <p className="text-sm font-serif italic">1024 x 1365 px</p>
            </div>
            <div>
              <p className="text-[10px] text-white/30 uppercase tracking-widest">Gemini Status</p>
              <p className="text-sm font-serif italic text-white/80">{isGenerating ? 'İşleniyor...' : 'Hazır (Ready)'}</p>
            </div>
          </div>

          {/* Main Output Frame */}
          <div className="w-full max-w-[600px] aspect-[3/4] bg-gradient-to-b from-[#121212] to-[#0A0A0A] rounded-sm shadow-2xl relative border border-white/[0.05] overflow-hidden flex items-center justify-center">
            
            <AnimatePresence mode="wait">
              {isGenerating && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-20 bg-[#050505]/70 backdrop-blur-xl flex flex-col items-center justify-center"
                >
                  <div className="relative">
                    <div className="w-16 h-16 border-t border-white/20 rounded-full animate-spin absolute inset-0"></div>
                    <div className="w-16 h-16 border-r border-white/80 rounded-full animate-spin absolute inset-0" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                  </div>
                  <p className="mt-8 text-[10px] uppercase tracking-[0.2em] text-white/50 font-serif italic">Stüdyo Renderlanıyor...</p>
                </motion.div>
              )}
            </AnimatePresence>

            {variations.length > 0 ? (
              <>
                <motion.img 
                  key={activeVariationIndex}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4 }}
                  src={variations[activeVariationIndex]} 
                  alt="Generated Try-On" 
                  className="w-full h-full object-contain md:object-cover border-none z-10 relative"
                />
                
                {/* Variations Toolbar */}
                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-30 flex items-center space-x-2 bg-[#0A0A0A]/95 backdrop-blur-xl p-2 rounded-xl border border-white/10 shadow-2xl">
                  {variations.map((v, i) => (
                    <button 
                      key={i} 
                      onClick={() => setActiveVariationIndex(i)} 
                      className={`w-10 h-14 rounded-md border ${
                        activeVariationIndex === i ? 'border-white' : 'border-transparent opacity-40 hover:opacity-100'
                      } overflow-hidden transition-all`}
                    >
                      <img src={v} alt={`Variation ${i+1}`} className="w-full h-full object-cover" />
                    </button>
                  ))}
                  
                  <div className="w-[1px] h-8 bg-white/20 mx-2"></div>
                  
                  <button 
                    onClick={() => handleGenerate(true)} 
                    disabled={isGenerating} 
                    className="w-10 h-14 rounded-md border border-dashed border-white/30 flex flex-col items-center justify-center hover:bg-white/10 transition-colors disabled:opacity-50 disabled:hover:bg-transparent group cursor-pointer"
                    title="Yeni Varyasyon Üret"
                  >
                    {isGenerating ? (
                      <Loader2 className="w-4 h-4 animate-spin text-white/50" />
                    ) : (
                      <span className="text-white/60 text-lg font-light group-hover:text-white transition-colors">+</span>
                    )}
                  </button>
                </div>
              </>
            ) : !isGenerating && (
              <div className="absolute inset-0 flex items-center justify-center text-center z-0">
                <div className="space-y-4 opacity-40">
                  <div className="w-12 h-[1px] bg-white mx-auto"></div>
                  <p className="font-serif italic text-lg text-white/60">Henüz bir önizleme oluşturulmadı.</p>
                  <p className="text-[10px] uppercase tracking-widest text-white/40">Görselleri yükleyip 'Oluştur'a tıklayın.</p>
                </div>
              </div>
            )}

            {/* Decorative Elements */}
            <div className="absolute top-4 left-4 z-20 text-[9px] text-white/30 tracking-tighter mix-blend-difference pointer-events-none">RENDER_V1.0_PRO_VISION</div>
            <div className="absolute bottom-4 right-4 z-20 text-[9px] text-white/30 tracking-tighter mix-blend-difference pointer-events-none">PROCESSED_BY_GEMINI_AI</div>
          </div>

        </section>
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
