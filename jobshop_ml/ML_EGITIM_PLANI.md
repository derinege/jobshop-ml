# ML Model Eğitim Planı

## 🎯 Projenin Amacı

**ML Model ile Job Shop Scheduling çözmek!**

## 📋 Eğitim Süreci

### Adım 1: Training Data Oluşturma (MIP ile)
- MIP solver ile küçük örnekler çözülür
- Her çözümden training sample'lar oluşturulur
- 100 training instance + 20 validation + 20 test

### Adım 2: GNN Modeli Eğitme
- Graph Neural Network eğitilir
- MIP çözümlerinden öğrenir (imitation learning)
- Model kaydedilir

### Adım 3: ML Modeli Kullanma
- Eğitilmiş model ile hızlı çözüm
- MIP'ten çok daha hızlı (saniyeler)

## ⏱️ Tahmini Süre

- **Training data oluşturma**: 2-4 saat (100 instance × ~2 dakika)
- **Model eğitimi**: 1-2 saat
- **Toplam**: 3-6 saat

## 🚀 Hızlı Başlangıç (Test için)

Küçük bir test için:
```bash
# Sadece 10 training instance ile test
python main.py --num-epochs 20 --batch-size 4
```

Tam eğitim için:
```bash
# Tam eğitim (100 instance, 100 epoch)
python main.py --num-epochs 100
```



