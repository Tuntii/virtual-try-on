/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Loader2, Download } from 'lucide-react';

interface PreviewAreaProps {
  variations: string[];
  activeIndex: number;
  isGenerating: boolean;
  onVariationSelect: (index: number) => void;
  onAddVariation: () => void;
}

export default function PreviewArea({
  variations,
  activeIndex,
  isGenerating,
  onVariationSelect,
  onAddVariation,
}: PreviewAreaProps) {
  const handleDownload = useCallback(() => {
    const src = variations[activeIndex];
    if (!src) return;
    const link = document.createElement('a');
    link.href = src;
    link.download = `virtual-try-on-${activeIndex + 1}.png`;
    link.click();
  }, [variations, activeIndex]);

  return (
    <section className="lg:col-span-8 xl:col-span-9 bg-[#050505] relative flex flex-col items-center justify-center p-6 lg:p-12 min-h-[60vh] lg:min-h-0 lg:h-full lg:overflow-hidden">
      {/* Top Info */}
      <div className="hidden lg:flex absolute top-8 left-8 space-x-12 z-10 pointer-events-none">
        <div>
          <p className="text-[10px] text-white/30 uppercase tracking-widest">Çözünürlük</p>
          <p className="text-sm font-serif italic">1024 x 1365 px</p>
        </div>
        <div>
          <p className="text-[10px] text-white/30 uppercase tracking-widest">Gemini Status</p>
          <p className="text-sm font-serif italic text-white/80">
            {isGenerating ? 'İşleniyor...' : 'Hazır (Ready)'}
          </p>
        </div>
      </div>

      {/* Desktop Download Button */}
      {variations.length > 0 && !isGenerating && (
        <button
          type="button"
          onClick={handleDownload}
          className="hidden lg:flex absolute top-8 right-8 z-10 items-center gap-2 px-4 py-2 border border-white/20 text-[10px] uppercase tracking-widest text-white/60 hover:text-white hover:border-white/50 transition-all"
        >
          <Download className="w-3 h-3" />
          <span>İndir</span>
        </button>
      )}

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
              <div className="relative w-16 h-16">
                <div className="w-16 h-16 border-t border-white/20 rounded-full animate-spin absolute inset-0" />
                <div
                  className="w-16 h-16 border-r border-white/80 rounded-full animate-spin absolute inset-0"
                  style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}
                />
              </div>
              <p className="mt-8 text-[10px] uppercase tracking-[0.2em] text-white/50 font-serif italic">
                Stüdyo Renderlanıyor...
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {variations.length > 0 ? (
          <>
            <motion.img
              key={activeIndex}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4 }}
              src={variations[activeIndex]}
              alt="Generated Try-On"
              className="w-full h-full object-contain md:object-cover border-none z-10 relative"
            />

            {/* Variations Toolbar */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-30 flex items-center space-x-2 bg-[#0A0A0A]/95 backdrop-blur-xl p-2 rounded-xl border border-white/10 shadow-2xl">
              {variations.map((v, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onVariationSelect(i)}
                  className={`w-10 h-14 rounded-md border ${
                    activeIndex === i
                      ? 'border-white'
                      : 'border-transparent opacity-40 hover:opacity-100'
                  } overflow-hidden transition-all`}
                  aria-label={`Varyasyon ${i + 1}`}
                >
                  <img src={v} alt={`Varyasyon ${i + 1}`} className="w-full h-full object-cover" />
                </button>
              ))}

              <div className="w-[1px] h-8 bg-white/20 mx-2" />

              <button
                type="button"
                onClick={onAddVariation}
                disabled={isGenerating}
                className="w-10 h-14 rounded-md border border-dashed border-white/30 flex flex-col items-center justify-center hover:bg-white/10 transition-colors disabled:opacity-50 disabled:hover:bg-transparent group cursor-pointer"
                title="Yeni Varyasyon Üret"
              >
                {isGenerating ? (
                  <Loader2 className="w-4 h-4 animate-spin text-white/50" />
                ) : (
                  <span className="text-white/60 text-lg font-light group-hover:text-white transition-colors">
                    +
                  </span>
                )}
              </button>
            </div>

            {/* Mobile Download Button */}
            <button
              type="button"
              onClick={handleDownload}
              className="lg:hidden absolute top-3 right-3 z-30 w-8 h-8 bg-black/70 rounded-full flex items-center justify-center hover:bg-black/90 transition-colors"
              aria-label="Görseli indir"
            >
              <Download className="w-4 h-4 text-white" />
            </button>
          </>
        ) : (
          !isGenerating && (
            <div className="absolute inset-0 flex items-center justify-center text-center z-0">
              <div className="space-y-4 opacity-40">
                <div className="w-12 h-[1px] bg-white mx-auto" />
                <p className="font-serif italic text-lg text-white/60">
                  Henüz bir önizleme oluşturulmadı.
                </p>
                <p className="text-[10px] uppercase tracking-widest text-white/40">
                  Görselleri yükleyip 'Oluştur'a tıklayın.
                </p>
              </div>
            </div>
          )
        )}

        {/* Decorative */}
        <div className="absolute top-4 left-4 z-20 text-[9px] text-white/30 tracking-tighter mix-blend-difference pointer-events-none">
          RENDER_V1.0_PRO_VISION
        </div>
        <div className="absolute bottom-4 right-4 z-20 text-[9px] text-white/30 tracking-tighter mix-blend-difference pointer-events-none">
          PROCESSED_BY_GEMINI_AI
        </div>
      </div>
    </section>
  );
}
