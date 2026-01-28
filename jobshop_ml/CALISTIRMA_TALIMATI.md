# Çalıştırma Talimatları

## ✅ Durum

- ✅ Excel dosyaları hazır: `jobshop_ml/` klasöründe
- ⚠️ Python paketleri yüklü değil

## 🚀 Çalıştırmak İçin

### 1. Paketleri Yükleyin

Terminal'de şu komutu çalıştırın:

```bash
cd /Users/derinegeevren/BIG_TEST/jobshop_ml
pip install pandas numpy openpyxl gurobipy tabulate
```

**Not**: Gurobi için lisans gerekebilir (akademik kullanıcılar için ücretsiz).

### 2. Çalıştırın

```bash
# Basit test
python main_scheduling.py --solve --export-excel test_schedule.xlsx

# Veya daha fazla seçenekle
python main_scheduling.py \
    --solve \
    --save-sol solution.pkl \
    --save-lp model.lp \
    --export-excel schedule.xlsx \
    --print-schedule \
    --time-limit 300
```

### 3. Sadece Veri Yükleme Testi (Gurobi olmadan)

Eğer Gurobi yüklü değilse, sadece veri yükleme testi yapabilirsiniz:

```python
from core import DataLoader, create_instance

loader = DataLoader()
loader.load_data()
print(f"BOLD jobs: {len(loader.get_bold_jobs())}")

instance = create_instance(loader)
print(f"Instance: {len(instance.jobs)} jobs, {len(instance.operations)} operations")
```

## 📋 Mevcut Dosyalar

✅ `islem_tam_tablo.xlsx` - jobshop_ml/ klasöründe
✅ `bold_islem_sure_tablosu.xlsx` - jobshop_ml/ klasöründe

## ⚠️ Sorun Giderme

### "ModuleNotFoundError: No module named 'pandas'"
**Çözüm**: `pip install pandas numpy openpyxl tabulate`

### "Gurobi license not found"
**Çözüm**: Gurobi lisansınızı ayarlayın veya sadece veri yükleme testi yapın

### "File not found"
**Çözüm**: Excel dosyalarının `jobshop_ml/` klasöründe olduğundan emin olun

## 🎯 Hızlı Test

```bash
# Hazırlık kontrolü
python hazirlik_kontrol.py

# Tam test
python test_refactored.py
```



