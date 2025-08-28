import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'

# 数据
num_ref = [200, 64, 32, 16, 8]
add_s = [95.8, 95.8, 94.7, 92.4, 87.2]
add_s_sym = [92.1, 92, 91.4, 90.4, 81.4]

plt.figure(figsize=(3.6, 2.8))  # 单栏大小，适合IEEE
plt.plot(num_ref, add_s, marker='o', label='ADD-S')
plt.plot(num_ref, add_s_sym, marker='s', label='ADD(-S)')

# 坐标轴
plt.xticks(num_ref, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(50, 100)

plt.xlabel('Number of reference images', fontsize=11)
plt.ylabel('AUC (%)', fontsize=11)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)  # 稍微小一点避免占空间

plt.tight_layout()

save_path = r"F:\gaussian-splatting\dataset\auc_curve.png"
plt.savefig(save_path, dpi=600, format='png')  # 用600dpi保证印刷清晰
plt.close()

print(f"已保存位图到: {save_path}")
