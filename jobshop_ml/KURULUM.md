# Kurulum ve Kullanıma Hazırlık

## ✅ Kod Durumu

**Kod tamamen hazır ve kullanıma uygun!** Tüm modüller oluşturuldu, test edildi ve dokümante edildi.

## 📦 Gerekli Paketler

Sistemi kullanmak için şu paketleri yüklemeniz gerekiyor:

```bash
pip install pandas numpy openpyxl gurobipy tabulate
```

**Not**: Gurobi için lisans gerekli (akademik kullanıcılar için ücretsiz).

## 🧪 Hızlı Test

Kurulumu test etmek için:

```bash
python test_refactored.py
```

Bu script:
- ✓ Bağımlılıkları kontrol eder
- ✓ Import'ları test eder
- ✓ Temel fonksiyonları test eder

## 🚀 Kullanıma Başlama

### 1. Paketleri Yükle

```bash
cd /Users/derinegeevren/BIG_TEST/jobshop_ml
pip install pandas numpy openpyxl gurobipy tabulate
```

### 2. Excel Dosyalarını Yerleştir

Bu dosyaları `jobshop_ml/` klasörüne koyun:
- `islem_tam_tablo.xlsx`
- `bold_islem_sure_tablosu.xlsx`

### 3. Test Et

```bash
# Test scripti çalıştır
python test_refactored.py

# Veya direkt kullan
python main_scheduling.py --solve --export-excel test_schedule.xlsx
```

## 📋 Özellikler

✅ **Modüler Yapı**: Tüm modüller `core/` paketinde ayrılmış
✅ **Optimize Edilmiş MIP**: Daha hızlı çözüm
✅ **Solution Save/Load**: Çözümleri kaydet/yükle
✅ **Excel Export**: Excel'e aktarım
✅ **Command-Line Interface**: Kolay kullanım
✅ **Jupyter Desteği**: Notebook'larda kullanılabilir

## 🔍 Sorun Giderme

### "ModuleNotFoundError: No module named 'pandas'"
**Çözüm**: `pip install pandas numpy openpyxl gurobipy tabulate`

### "Gurobi license not found"
**Çözüm**: Gurobi lisansınızı ayarlayın (akademik kullanıcılar için ücretsiz)

### "File not found: islem_tam_tablo.xlsx"
**Çözüm**: Excel dosyalarını `jobshop_ml/` klasörüne koyun

## 📚 Dokümantasyon

- `REFACTORING_SUMMARY.md`: Detaylı refactoring özeti
- `QUICK_REFERENCE.md`: Hızlı referans kılavuzu
- `START_HERE.txt`: Başlangıç kılavuzu

## ✨ Örnek Kullanım

```python
from core import DataLoader, create_instance, ModelBuilder, Solver, ExcelWriter

# Veri yükle
loader = DataLoader()
loader.load_data()

# Instance oluştur
instance = create_instance(loader)

# Model oluştur ve çöz
builder = ModelBuilder()
model = builder.build_model(instance)
solver = Solver(time_limit=300)
solution = solver.solve(model, instance)

# Kaydet ve Excel'e aktar
solution.save('solution.pkl')
excel_writer = ExcelWriter()
excel_writer.export_schedule(solution, 'schedule.xlsx')
```

## 🎯 Sonuç

**Kod %100 hazır!** Sadece paketleri yüklemeniz ve Excel dosyalarınızı yerleştirmeniz yeterli.



