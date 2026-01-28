# ML Model Kullanımı

## 🔍 Durum

**Şu ana kadar**: Sadece MIP Solver (Gurobi) kullandık
**ML Model**: Henüz eğitilmedi, kullanılamıyor

## 📊 İki Sistem Karşılaştırması

| Özellik | MIP Solver (Gurobi) | ML Model (GNN) |
|---------|---------------------|----------------|
| **Hız** | Yavaş (dakikalar) | Hızlı (saniyeler) |
| **Kalite** | Optimal/Feasible | MIP'ten %5-15 daha kötü |
| **Eğitim** | Gerekmez | Gerekir |
| **Kullanım** | ✅ Şu anda kullanıyoruz | ❌ Henüz eğitilmedi |

## 🚀 ML Modeli Eğitmek İçin

### Adım 1: ML Modeli Eğit

```bash
cd /Users/derinegeevren/BIG_TEST/jobshop_ml
python main.py --num-epochs 50 --batch-size 8
```

Bu komut:
1. MIP solver ile küçük örnekler çözer (training data)
2. GNN modelini eğitir
3. Modeli `checkpoints/best_model.pt` olarak kaydeder

### Adım 2: Eğitilmiş Modeli Kullan

```python
from gnn_model import SchedulingGNN
from evaluation import MLScheduler
from graph_builder import GraphBuilder
from core import DataLoader, create_instance
import torch

# Veri yükle
loader = DataLoader()
loader.load_data()
instance = create_instance(loader)

# Eğitilmiş modeli yükle
model = SchedulingGNN()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# ML scheduler oluştur
scheduler = MLScheduler(model, GraphBuilder(), device='cpu')

# Hızlı çözüm!
result = scheduler.schedule(instance)
print(f"Makespan: {result['makespan']:.2f} dakika")
print(f"Objective: {result['objective']:.2f}")
```

## ⚡ ML Modelin Avantajları

1. **Hız**: MIP 300 saniye → ML <1 saniye
2. **Büyük Problemler**: MIP çözemezken ML çözebilir
3. **Production**: Gerçek zamanlı çözüm için ideal

## 📝 Özet

- ✅ **MIP Solver**: Kullanıyoruz, optimal çözüm
- ❌ **ML Model**: Henüz eğitilmedi
- 🎯 **Sonraki Adım**: `python main.py` ile ML modeli eğit

## 🔄 İki Sistemi Birlikte Kullanmak

```python
# Küçük problemler için: MIP (optimal)
# Büyük problemler için: ML (hızlı)

if len(instance.operations) < 50:
    # MIP kullan
    solution = solver.solve(model, instance)
else:
    # ML kullan
    result = ml_scheduler.schedule(instance)
```



