import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

file_path = r'D:\some_scripts\learn_2024_01\CCL\CFN2.1\frame_info.json'

# with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         count = 0
#         for data_item in data:
#             count += 1
#             print(f"Frame {count}: {data_item['frame_name']}")


with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    count = 0
    for data_item in data:
        frame_name = data_item['frame_name']
        fes_count = len(data_item['fes'])
        count += 1
        print(f"框架 {frame_name} 有 {fes_count} 个框架元素")
        print(f"一共有{count}个框架")




# 确保matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有的fes中的fe_name作为统计对象
all_fes = [fe['fe_name'] for item in data for fe in item['fes']]

# 进行词频统计
fes_counts = Counter(all_fes)

# 获取最常见的10个fes及其频率
common_fes = fes_counts.most_common(70)
fes, frequencies = zip(*common_fes)

# 可视化
plt.figure(figsize=(70, 8))
plt.bar(fes, frequencies, color='skyblue')
plt.xlabel('框架元素')
plt.ylabel('频率')
plt.title('Top 70 框架元素频率')
plt.xticks(rotation=45)

# 保存图像
# plt.savefig('Top_10_FES_Frequency.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
