import pandas as pd

# ========== 配置 ==========
METADATA_PATH = r"E:\GeoFair-Building-v1.0\metadata.csv"
# ==========================

# 读取元数据
df = pd.read_csv(METADATA_PATH)

# 修复 hdi_level
# 北美洲和欧洲的国家都是 developed
df.loc[df['continent'] == 'North America', 'hdi_level'] = 'developed'
df.loc[df['continent'] == 'Europe', 'hdi_level'] = 'developed'
df.loc[df['continent'] == 'Oceania', 'hdi_level'] = 'developed'
df.loc[df['continent'] == 'Asia', 'hdi_level'] = 'developing'
df.loc[df['continent'] == 'South America', 'hdi_level'] = 'developing'
df.loc[df['continent'] == 'Africa', 'hdi_level'] = 'underdeveloped'

# 检查是否还有 Unknown
unknown = df[df['hdi_level'] == 'Unknown']
if len(unknown) > 0:
    print(f"警告: 仍有 {len(unknown)} 个 Unknown 样本")
    print(unknown[['continent', 'source', 'hdi_level']].head())
else:
    print("✅ 所有 hdi_level 已修复")

# 保存修复后的文件
df.to_csv(METADATA_PATH, index=False)

# 输出修复后的分布
print("\n修复后的 HDI 等级分布:")
print(df['hdi_level'].value_counts())

print("\n大洲与 HDI 交叉表:")
print(pd.crosstab(df['continent'], df['hdi_level']))