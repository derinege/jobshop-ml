# Excel Dosyalarını Nereye Koymalıyım?

## 📁 Doğru Konum

Excel dosyalarınızı **şu klasöre** koyun:

```
/Users/derinegeevren/BIG_TEST/jobshop_ml/
```

Yani `jobshop_ml` klasörünün **içine** direkt olarak.

## 📋 Gerekli Dosyalar

Bu iki dosyayı `jobshop_ml/` klasörüne koyun:

1. ✅ `islem_tam_tablo.xlsx`
2. ✅ `bold_islem_sure_tablosu.xlsx`

## ✅ Doğru Yerleşim

```
jobshop_ml/
├── islem_tam_tablo.xlsx          ← BURAYA
├── bold_islem_sure_tablosu.xlsx  ← BURAYA
├── config.py
├── main_scheduling.py
├── core/
│   ├── ...
└── ...
```

## 🔍 Kontrol Etme

Dosyaların doğru yerde olup olmadığını kontrol etmek için:

```bash
cd /Users/derinegeevren/BIG_TEST/jobshop_ml
ls -la *.xlsx
```

Bu komut şunları göstermeli:
- `islem_tam_tablo.xlsx`
- `bold_islem_sure_tablosu.xlsx`

## ⚙️ Farklı Bir Yere Koymak İsterseniz

Eğer dosyaları başka bir yere koymak isterseniz, `config.py` dosyasını düzenleyin:

```python
# config.py içinde
DATA_PATH_ISLEM_TAM = "/tam/yol/islem_tam_tablo.xlsx"
DATA_PATH_BOLD_SURE = "/tam/yol/bold_islem_sure_tablosu.xlsx"
```

Veya komut satırından:

```bash
python main_scheduling.py \
    --islem-tam-path /tam/yol/islem_tam_tablo.xlsx \
    --bold-sure-path /tam/yol/bold_islem_sure_tablosu.xlsx \
    --solve
```

## ✅ Test

Dosyalar doğru yerdeyse, şu komut çalışmalı:

```bash
python main_scheduling.py --solve
```

Eğer "File not found" hatası alırsanız, dosyalar yanlış yerde demektir.

## 📝 Özet

**Kısa cevap**: Excel dosyalarını `jobshop_ml/` klasörünün **içine** direkt koyun!



