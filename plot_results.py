import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Lasso Regression', 'Linear Regression']
mae = [1.73, 2.25, 2.96, 2.96, 2.96]
r2 = [0.9301, 0.9154, 0.8628, 0.8627, 0.8628]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(models, mae, color=['steelblue', 'lightblue', 'lightcoral', 'lightcoral', 'lightcoral'])
ax1.set_xlabel('Model')
ax1.set_ylabel('MAE (minutes)')
ax1.set_title('Mean Absolute Error by Model')
ax1.tick_params(axis='x', rotation=45)
for bar, val in zip(bars1, mae):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val}', ha='center', fontweight='bold')

bars2 = ax2.bar(models, r2, color=['steelblue', 'lightblue', 'lightcoral', 'lightcoral', 'lightcoral'])
ax2.set_xlabel('Model')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score by Model')
ax2.tick_params(axis='x', rotation=45)
for bar, val in zip(bars2, r2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Chart saved as 'model_comparison.png'")