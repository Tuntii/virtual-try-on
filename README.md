# Virtual Try-On POC

Tekstil demoları için `Streamlit` tabanlı, **tek ürün + çoklu manken** senaryosunu destekleyen POC. İki backend vardır:

- **`cpu-composite`** (varsayılan, her makinede çalışır): `rembg` + OpenCV ile ürünü torso hattına yerleştirir. Hızlı demo içindir; gerçek generative değildir.
- **`gpu-inpaint`** (CUDA GPU gerektirir): Stable Diffusion Inpainting + IP-Adapter ile torso bölgesini, ürün görselinin stilini referans alarak yeniden üretir. `requirements-gpu.txt` kurulduktan ve `torch` CUDA ile geldiğinde sidebar'da otomatik aktifleşir.

> Bu bir POC'tur. `gpu-inpaint` backend stil/renk/desen aktarımı yapar; ürünü piksel düzeyinde birebir klonlayan gerçek bir VTON modeli değildir. Garment-exact sonuç için ileride CatVTON/IDM-VTON entegrasyonu önerilir (bkz. `generative_stub.py`).

## Kurulum (Windows, Python 3.12 önerilir)

```powershell
cd c:\Users\tunay\Documents\GitHub\virtual-try-on
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU backend (opsiyonel, CUDA 12.x + NVIDIA GPU)

```powershell
# CUDA-enabled torch (6 GB+ VRAM önerilir)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
# Diffusers + IP-Adapter bağımlılıkları
pip install -r requirements-gpu.txt
```

İlk çalıştırmada `runwayml/stable-diffusion-inpainting` (~2 GB) ve `h94/IP-Adapter` ağırlıkları Hugging Face'ten indirilir. 6 GB VRAM'li kartlar (ör. RTX 2060) için uygulama fp16, attention/VAE slicing ve 512 px kısa kenar ile çalışacak şekilde yapılandırılmıştır.

## Çalıştırma

```powershell
streamlit run app.py
```

Tarayıcı otomatik açılmazsa `http://localhost:8501` adresine gidin.

## Girdi kalitesi (önemli)

Sonucun yarısı modelde, diğer yarısı input kalitesinde biter. İdeal kurulum:

- **Ürün görseli**: Tek parça packshot, düz/şeffaf zemin, ön cepheden, el/kol/insan görünmüyor, kare-ya-kın oran, ≥ 1000 px.
- **Manken fotoğrafı**: Ön cepheli, tam gövde görünüyor, sade arka plan, ≥ 512 px kısa kenar, aşırı yatay değil.

Uygulama yüklediğiniz görselleri analiz eder ve uygunsuz input'lar için "Kalite uyarıları" bölümünde rehberlik eder.

## Kullanım

1. **Ürün görseli** seçin (PNG + şeffaf arka plan idealdir; düz fonlu JPG de otomatik temizlenir).
2. Bir veya birden fazla **manken fotoğrafı** yükleyin (ön cepheli, tam gövde en iyi sonucu verir).
3. Sol panelden backend'i seçin (GPU varsa `gpu-inpaint` önerilir) ve ayarları ince ayarlayın.
4. **Denemeyi çalıştır**'a basın. Sonuçları galeri görünümünde inceleyin ve PNG olarak indirin.

## Örnek görseller

`assets/samples/garments/` ve `assets/samples/models/` klasörlerine telifsiz örnek görseller koyun; uygulama bunları "Örneklerden seç" sekmesinde otomatik listeler.

## Klasör yapısı

```
app.py                          # Streamlit uygulaması
requirements.txt
requirements-gpu.txt            # Opsiyonel GPU backend bağımlılıkları
src/
  core/                         # Veri sözleşmeleri, hata tipleri
  backends/
    base.py                     # TryOnBackend arayüzü
    cpu_composite_backend.py
    gpu_inpaint_backend.py      # SD Inpainting + IP-Adapter (CUDA)
    generative_stub.py          # İleride CatVTON/IDM-VTON entegrasyonu için iskelet
  services/
    image_io.py                 # Yükleme, EXIF, resize, doğrulama
    pose_estimator.py           # Silüet tabanlı omuz/kalça tahmini
    person_parser.py            # rembg ile kişi silüeti
    garment_mask.py             # rembg + alfa kırpma
    torso_mask.py               # SD inpaint için torso maskesi
    quality_gate.py             # Girdi kalite kontrolleri
    device.py                   # CUDA/torch tespiti
    catalog.py                  # Örnek görsel listesi
  ui/
    components.py               # Sonuç kartı ve galeri
assets/
  samples/
    garments/
    models/
```

## Bilinen limitler

- Yalnızca üst beden (tişört/gömlek/ceket) için ayarlanmıştır; alt beden/elbise için ek torso maskesi ve parametreleme gerekir.
- `gpu-inpaint` IP-Adapter ile **stil + renk + desen aktarımı** yapar; ürünü piksel düzeyinde birebir klonlayan gerçek bir VTON modeli değildir.
- 6 GB VRAM sınırı nedeniyle çalışma çözünürlüğü 512 px kısa kenara sabittir.
- Yan profil veya aşırı yakın çekim fotoğraflarda silüet tabanlı torso tahmini zayıflar.

## İleriye dönük

- `src/backends/generative_stub.py` içine **CatVTON** veya **IDM-VTON** entegre edilerek garment-exact "yüksek kalite modu" eklenebilir. UI aynı kalır.
- WSL2 + Linux torch kurulumu Windows native'e göre daha kararlıdır; kalite hassasiyeti yüksek senaryolar için önerilir.
