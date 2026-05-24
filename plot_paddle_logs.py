import re
import matplotlib.pyplot as plt
import numpy as np

log_file = r"D:\Computer Vision Project\models\best_accuracy\train.log"

epoch_loss_dict = {}

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    # Match training log: epoch: [5/100], global_step: 490... loss: 28.421073
    train_match = re.search(r'epoch: \[(\d+)/\d+\].*?loss: ([\d\.]+)', line)
    if train_match:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        
        if epoch not in epoch_loss_dict:
            epoch_loss_dict[epoch] = []
        epoch_loss_dict[epoch].append(loss)

# Calculate average loss per epoch
epochs = sorted(list(epoch_loss_dict.keys()))
avg_losses = [np.mean(epoch_loss_dict[e]) for e in epochs]

plt.figure(figsize=(10, 6))

# Plot only Train Loss vs Epoch
plt.plot(epochs, avg_losses, label='Train Loss', color='#FF5733', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
output_path = r"D:\Computer Vision Project\training_logs_chart.png"
plt.savefig(output_path, dpi=300)
print(f"Chart successfully saved to: {output_path}")
