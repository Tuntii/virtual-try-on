/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useState, useCallback } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

const MAX_FILE_SIZE_MB = 4;

interface ImageUploadProps {
  step: string;
  label: string;
  image: string | null;
  mime: string | null;
  placeholder: string;
  hint: string;
  onFile: (base64: string, mime: string) => void;
  onClear: () => void;
}

export default function ImageUpload({
  step,
  label,
  image,
  mime,
  placeholder,
  hint,
  onFile,
  onClear,
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [sizeError, setSizeError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File) => {
      setSizeError(null);
      if (!file.type.startsWith('image/')) {
        setSizeError('Lütfen bir görsel dosyası seçin.');
        return;
      }
      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        setSizeError(`Dosya boyutu ${MAX_FILE_SIZE_MB}MB'ı geçemez.`);
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        const [header, base64] = result.split(',');
        const fileMime = header.replace('data:', '').replace(';base64', '');
        onFile(base64, fileMime);
      };
      reader.readAsDataURL(file);
    },
    [onFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
      // Reset input so same file can be re-selected
      e.target.value = '';
    },
    [processFile]
  );

  return (
    <section>
      <label className="text-[10px] uppercase tracking-widest text-white/40 mb-3 block">
        {step}. {label}
      </label>
      <div
        role="button"
        tabIndex={0}
        aria-label={`${label} yükle`}
        className={`relative w-full aspect-square md:aspect-video lg:aspect-square border border-dashed rounded-lg flex flex-col items-center justify-center bg-white/[0.02] cursor-pointer overflow-hidden group transition-colors ${
          isDragging
            ? 'border-white/60 bg-white/[0.08]'
            : 'border-white/20 hover:bg-white/[0.05]'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />
        {image && mime ? (
          <>
            <img
              src={`data:${mime};base64,${image}`}
              alt={label}
              className="w-full h-full object-cover"
            />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setSizeError(null);
                onClear();
              }}
              className="absolute top-2 right-2 w-6 h-6 bg-black/70 rounded-full flex items-center justify-center hover:bg-black/90 transition-colors z-10"
              aria-label="Görseli kaldır"
            >
              <X className="w-3 h-3 text-white" />
            </button>
          </>
        ) : (
          <>
            {isDragging ? (
              <Upload className="w-6 h-6 mb-2 text-white/80" strokeWidth={1.5} />
            ) : (
              <ImageIcon className="w-6 h-6 mb-2 text-white/30" strokeWidth={1.5} />
            )}
            <span
              className={`text-[11px] italic font-serif transition-colors ${
                isDragging ? 'text-white/80' : 'text-white/30 group-hover:text-white/50'
              }`}
            >
              {isDragging ? 'Bırak!' : placeholder}
            </span>
            {!isDragging && (
              <span className="text-[10px] text-white/20 mt-1">ya da sürükle & bırak</span>
            )}
          </>
        )}
      </div>
      {sizeError ? (
        <div className="mt-2 text-[10px] text-red-400">{sizeError}</div>
      ) : (
        <div className="mt-2 text-[10px] text-white/40">{hint}</div>
      )}
    </section>
  );
}
