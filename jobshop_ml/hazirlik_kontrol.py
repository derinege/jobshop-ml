#!/usr/bin/env python3
"""
Hazırlık kontrol scripti - çalıştırmadan önce kontrol edin
"""

import sys
import os

def check_excel_files():
    """Excel dosyalarını kontrol et"""
    print("📊 Excel Dosyaları Kontrolü:")
    print("-" * 50)
    
    files = [
        "islem_tam_tablo.xlsx",
        "bold_islem_sure_tablosu.xlsx"
    ]
    
    all_ok = True
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024  # KB
            print(f"  ✓ {f} ({size:.1f} KB)")
        else:
            print(f"  ✗ {f} BULUNAMADI")
            all_ok = False
    
    return all_ok

def check_packages():
    """Python paketlerini kontrol et"""
    print("\n📦 Python Paketleri Kontrolü:")
    print("-" * 50)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'openpyxl': 'openpyxl',
        'gurobipy': 'gurobipy',
        'tabulate': 'tabulate'
    }
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} YÜKLÜ DEĞİL")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 50)
    print("HAZIRLIK KONTROLÜ")
    print("=" * 50)
    print()
    
    excel_ok = check_excel_files()
    packages_ok = check_packages()
    
    print("\n" + "=" * 50)
    print("SONUÇ")
    print("=" * 50)
    
    if excel_ok and packages_ok:
        print("✅ HER ŞEY HAZIR! Çalıştırabilirsiniz:")
        print()
        print("  python main_scheduling.py --solve --export-excel schedule.xlsx")
        return 0
    else:
        print("⚠️  EKSİKLER VAR:")
        print()
        if not excel_ok:
            print("  • Excel dosyalarını jobshop_ml/ klasörüne koyun")
            print("    - islem_tam_tablo.xlsx")
            print("    - bold_islem_sure_tablosu.xlsx")
        if not packages_ok:
            print("  • Paketleri yükleyin:")
            print("    pip install pandas numpy openpyxl gurobipy tabulate")
        return 1

if __name__ == "__main__":
    sys.exit(main())



