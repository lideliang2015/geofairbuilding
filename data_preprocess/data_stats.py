"""
GeoFair-Building v1.0 数据统计脚本
统计最终测试集的分布情况
"""

import pandas as pd
import os

# ========== 配置 ==========
METADATA_PATH = r"E:\GeoFair-Building-v1.0\metadata.csv"
OUTPUT_DIR = r"E:\GeoFair-Building-v1.0"
# ==========================

def main():
    print("=" * 60)
    print("GeoFair-Building v1.0 数据统计")
    print("=" * 60)
    
    # 读取元数据
    df = pd.read_csv(METADATA_PATH)
    
    print(f"\n总瓦片数: {len(df)}")
    
    # 1. 大洲分布
    print("\n" + "-" * 40)
    print("【1. 大洲分布】")
    print("-" * 40)
    continent_counts = df['continent'].value_counts()
    for cont, count in continent_counts.items():
        pct = count / len(df) * 100
        print(f"  {cont}: {count} ({pct:.1f}%)")
    
    # 2. HDI 等级分布
    print("\n" + "-" * 40)
    print("【2. HDI 等级分布】")
    print("-" * 40)
    hdi_counts = df['hdi_level'].value_counts()
    for hdi, count in hdi_counts.items():
        pct = count / len(df) * 100
        print(f"  {hdi}: {count} ({pct:.1f}%)")
    
    # 3. 数据源分布
    print("\n" + "-" * 40)
    print("【3. 数据源分布】")
    print("-" * 40)
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        pct = count / len(df) * 100
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    # 4. 分辨率分布
    print("\n" + "-" * 40)
    print("【4. 分辨率分布】")
    print("-" * 40)
    if 'resolution_m' in df.columns:
        res_counts = df['resolution_m'].value_counts().sort_index()
        for res, count in res_counts.items():
            pct = count / len(df) * 100
            print(f"  {res}m: {count} ({pct:.1f}%)")
    else:
        print("  无分辨率信息")
    
    # 5. 街区类型分布（如果有）
    print("\n" + "-" * 40)
    print("【5. 街区类型分布】")
    print("-" * 40)
    if 'settlement_type' in df.columns:
        type_counts = df['settlement_type'].value_counts()
        for typ, count in type_counts.items():
            pct = count / len(df) * 100
            print(f"  {typ}: {count} ({pct:.1f}%)")
    else:
        print("  无街区类型信息")
    
    # 6. 大洲 × HDI 交叉表
    print("\n" + "-" * 40)
    print("【6. 大洲 × HDI 交叉表】")
    print("-" * 40)
    cross = pd.crosstab(df['continent'], df['hdi_level'])
    print(cross)
    
    # 7. 保存统计报告
    print("\n" + "-" * 40)
    print("【7. 保存统计报告】")
    print("-" * 40)
    
    report_path = os.path.join(OUTPUT_DIR, "statistics_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GeoFair-Building v1.0 统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总瓦片数: {len(df)}\n\n")
        f.write("大洲分布:\n")
        f.write(continent_counts.to_string() + "\n\n")
        f.write("HDI 等级分布:\n")
        f.write(hdi_counts.to_string() + "\n\n")
        f.write("数据源分布:\n")
        f.write(source_counts.to_string() + "\n\n")
        f.write("大洲 × HDI 交叉表:\n")
        f.write(cross.to_string() + "\n")
    
    print(f"统计报告已保存至: {report_path}")
    
    print("\n" + "=" * 60)
    print("统计完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()