# 🚗 Araç Takip, Sayım ve Plaka Tanıma Sistemi

> **Önemli not:** Video dosyasını GitHub deposuna ekleyemediğim için `.mp4` dosyası bu repoda bulunmayabilir. Projeyi çalıştırmak için videonuzu `input_videos/arac_giris.mp4` konumuna yerleştirin. Dosya bu yolda bulunduğunda ve `config.py` içindeki yol doğru olduğunda kod videoyu işlemeye hazırdır.

GitHub deposu: [Skylowerr/arac_takip](https://github.com/Skylowerr/arac_takip)

Bu proje, video görüntüsü üzerinde araçları tespit etmek, araçları kareler arasında takip etmek, sanal bir çizgiden geçen araçları saymak ve araç plakalarını OCR ile okumak amacıyla hazırlanmıştır. Plaka görüntüleri iyileştirilmiş sürümleriyle birlikte kaydedilir; geçiş bilgileri CSV, JSON ve TXT formatlarında dışa aktarılır.

## Özellikler

- **Araç tespiti:** YOLOv8x ile otomobil, motosiklet, otobüs ve kamyon algılama.
- **Araç takibi:** ByteTrack kullanarak kareler arasında takip ID'lerini korumaya çalışma.
- **Sanal çizgiyle sayım:** Görüntü boyunca uzanan çizgi üzerinden IN ve OUT geçişlerini sayma.
- **Plaka tespiti:** Hazır YOLO tabanlı plaka tespit modeliyle araç içindeki plaka bölgesini bulma.
- **Plaka okuma (OCR):** EasyOCR ile plaka üzerindeki Latin harflerini ve rakamları okuma.
- **Görüntü iyileştirme:** Plaka kırpımlarını büyütme, kontrast artırma ve keskinleştirme.
- **Video üstü bilgi:** Araç kutuları, araç ID'si, araç türü, bilinen plaka, anlık araç sayısı, IN/OUT sayaçları ve işleme FPS'i.
- **Dışa aktarma:** CSV, JSON ve TXT olay kayıtları.
- **Görsel kayıtları:** Araç kırpımları ve orijinal/iyileştirilmiş plaka görselleri ayrı klasörlere yazılır.
- **Yabancı plakalar:** Türkiye plaka biçimi zorunlu tutulmaz. Okunan karakterler metin olarak kaydedilir; plakanın ülkesini OCR tek başına kesin belirlemez.

## Proje yapısı

```text
arac_takip/
├── input_videos/
│   └── arac_giris.mp4          # Kullanıcının eklemesi gereken video
├── outputs/                    # Çalıştırma sırasında otomatik oluşturulur
│   ├── output_results_full.mp4
│   ├── vehicle_events.csv
│   ├── vehicle_events.json
│   ├── vehicle_events.txt
│   ├── vehicle_crops/          # Araç görüntüleri
│   └── plate_crops/
│       ├── original/           # Orijinal plaka kırpımları
│       └── enhanced/            # İyileştirilmiş plaka kırpımları
├── config.py                   # Dosya yolları, model ve eşik ayarları
├── main.py                     # Ana video işleme döngüsü
├── plate_reader.py             # Plaka tespitiyle birlikte kullanılan OCR ve iyileştirme yardımcıları
├── utils.py                    # FPS paneli, IN/OUT çizimi ve zaman biçimi
├── results.py                  # CSV, JSON ve TXT dışa aktarma
├── requirements.txt            # Python paketleri
└── README.md
```

> `outputs/` içindeki dosyalar video işlendiğinde oluşturulur. Bu klasörün başlangıçta boş olması normaldir.

## Gereksinimler

- Python 3.10 veya uyumlu bir Python sürümü
- Google Colab ya da yeterli belleğe sahip yerel bir bilgisayar
- GPU önerilir; GPU yoksa kod CPU ile çalışabilir ancak daha yavaş olabilir
- İnternet bağlantısı: YOLO ağırlıkları ve plaka tespit modelinin ilk kez indirilmesi için gerekir

## Kurulum

### 1. Depoyu indirin

```bash
git clone https://github.com/Skylowerr/arac_takip.git
cd arac_takip
```

### 2. Videoyu doğru konuma yerleştirin

Video dosyasının yolu şu olmalıdır:

```text
input_videos/arac_giris.mp4
```

Kendi dosyanızın adı farklıysa `config.py` içindeki `INPUT_VIDEO` değerini güncelleyin.

Örnek:

```python
INPUT_VIDEO = Path("input_videos/arac_giris.mp4")
```

### 3. Paketleri yükleyin

```bash
python -m pip install -r requirements.txt
```

Google Colab kullanıyorsanız bir kod hücresinde şu komutu çalıştırabilirsiniz:

```python
%pip install -r requirements.txt
```

Kurulumdan sonra gerekli olursa Colab oturumunu yeniden başlatın.

### 4. Videoyu çalıştırın

Proje klasöründe şu komutu yürütün:

```bash
python main.py
```

Kod, kaynak videonun sonuna kadar okur. İşleme süresi video uzunluğuna, çözünürlüğe ve kullanılan donanıma göre değişir.

## Önemli ayarlar

Ayarların çoğu `config.py` içindedir.

| Ayar | Açıklama |
|---|---|
| `INPUT_VIDEO` | İşlenecek `.mp4` dosyasının yolu |
| `OUTPUT_VIDEO` | İşlenmiş videonun kaydedileceği yol |
| `LINE_RATIO` | Sanal sayım çizgisinin görüntü yüksekliğine oranı. Örneğin `0.70`, çizgiyi görüntünün üstünden %70 aşağıya yerleştirir. |
| `VEHICLE_CONF` | Araç tespiti güven eşiği. Düşürmek daha fazla araç yakalayabilir ama yanlış tespitleri artırabilir. |
| `VEHICLE_IMGSZ` | Araç tespitinde kullanılan giriş boyutu. Artırmak küçük/uzak araçlara yardımcı olabilir fakat işlemi yavaşlatır. |
| `PLATE_CONF` | Plaka kutusu için tespit güven eşiği |
| `PLATE_CHECK_INTERVAL` | Aynı takip edilen araç için plaka tespiti yapılmadan önce geçmesi gereken kare sayısı |

**IN/OUT yönü:** Kodun varsayılanında yukarıdan aşağıya geçiş `IN`, aşağıdan yukarıya geçiş `OUT` kabul edilir. Videodaki trafik yönü farklıysa `main.py` içindeki yön kontrolü buna göre değiştirilmelidir.

## Üretilen dosyalar

İşlem tamamlandığında `outputs/` altında aşağıdaki çıktılar bulunur:

- `output_results_full.mp4`: Araç kutuları, plaka metinleri, sanal çizgi, IN/OUT sayaçları ve FPS göstergesi eklenmiş video.
- `vehicle_events.csv`: Excel veya pandas ile incelenebilen geçiş tablosu.
- `vehicle_events.json`: Yapısal olay kayıtları.
- `vehicle_events.txt`: Okunabilir metin raporu.
- `vehicle_crops/`: Geçiş anında kaydedilen araç görüntüleri.
- `plate_crops/original/`: Modelin bulduğu plakanın orijinal kırpımı.
- `plate_crops/enhanced/`: Büyütülmüş ve görüntü iyileştirme uygulanmış plaka kırpımı.

Olay kayıtlarında, kodun geçerli sürümüne bağlı olarak zaman, kare numarası, araç ID'si, araç türü, geçiş yönü, araç tespit güveni, plaka metni, plaka tespit güveni ve OCR güveni tutulur.

## Kullanılan teknolojiler

- **Python:** Proje dili.
- **OpenCV:** Video okuma/yazma, çizimler, kırpımlar ve görüntü iyileştirme.
- **Ultralytics YOLOv8x:** COCO sınıflarından araç algılama.
- **ByteTrack:** Kareler arasında araç takibi.
- **YOLO tabanlı plaka tespit modeli:** Hugging Face model deposundan indirilen `best.pt` ağırlığı.
- **EasyOCR:** Plaka üzerindeki karakterleri metne dönüştürme.
- **NumPy / PyTorch:** Sayısal işlemler ve GPU desteği.

İlk çalıştırmada model ağırlıkları indirilebilir. Model adları, lisans ve kullanım koşulları için ilgili model deposu ve kütüphanelerin belgeleri kontrol edilmelidir.

## Sınırlamalar ve notlar

- Plaka tespiti ve OCR doğruluğu; plakanın görüntüdeki boyutuna, açısına, hareket bulanıklığına, ışığa ve kapanma durumuna bağlıdır.
- Görüntü büyütme ve kontrast iyileştirme, kameranın hiç kaydetmediği harf ayrıntılarını geri getiremez. Okunan plakaları önemli bir kullanım öncesinde insan tarafından doğrulayın.
- Takip ID'leri uzun örtülmelerde değişebilir. Bu durumda tek aracın birden fazla kez sayılması veya bazı geçişlerin kaçırılması mümkündür.
- Sanal çizginin doğru yerde olması ve IN/OUT yönlerinin videoya uygun ayarlanması gerekir.
- OCR'ın okuduğu metin, plakanın yabancı olduğunu kesin kanıtlamaz. Ülke tespiti bu projede ayrıca doğrulanmıyorsa sonuçlar yalnızca metin okuma sonucu olarak değerlendirilmelidir.
- Kod, ilk çalıştırmada model indirmek için internet bağlantısına ihtiyaç duyabilir.

## Sorun giderme

**`Video bulunamadı` hatası:** `input_videos/arac_giris.mp4` dosyasının mevcut olduğunu ve dosya adının `config.py` ile aynı olduğunu kontrol edin.

**GPU kullanılmıyor:** Colab'da `Runtime → Change runtime type → T4 GPU` (veya mevcut bir GPU seçeneği) ayarlanabilir. GPU kullanılmasa da CPU ile çalışabilir; daha yavaş olacaktır.

**Plaka okunamıyor:** `outputs/plate_crops/original/` ve `outputs/plate_crops/enhanced/` klasörlerindeki fotoğrafları karşılaştırın. Plaka tespit modeli plakayı yanlış kırpıyorsa yalnızca OCR ayarlarını değiştirmek yeterli olmayabilir.

**Bağımlılık çakışması:** Paketleri projenin `requirements.txt` dosyasından kurun. Kurulumdan sonra Colab oturumunu yeniden başlatmak gerekebilir.

---

Bu proje bilgisayarlı görü, nesne takibi ve OCR yöntemlerini öğrenmek ve denemek amacıyla hazırlanmıştır. Sayım ve plaka okuma sonuçları kesin doğruluk garantisi vermez.
