#!/bin/bash
# Çalıştırma scripti - Anaconda Python kullanır

# Anaconda Python'u kullan
PYTHON="/opt/anaconda3/bin/python"

echo "🚀 Job Shop Scheduling Optimizer"
echo "=================================="
echo ""

# Veri yükleme testi
echo "📊 Veri yükleme testi..."
$PYTHON -c "
from core import DataLoader, create_instance
loader = DataLoader()
loader.load_data()
print(f'✓ {len(loader.get_bold_jobs())} BOLD job bulundu')
instance = create_instance(loader)
print(f'✓ Instance: {len(instance.jobs)} jobs, {len(instance.operations)} operations')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Veri yükleme başarılı!"
    echo ""
    echo "Çalıştırmak için:"
    echo "  $PYTHON main_scheduling.py --solve --export-excel schedule.xlsx"
else
    echo ""
    echo "❌ Hata oluştu"
fi



