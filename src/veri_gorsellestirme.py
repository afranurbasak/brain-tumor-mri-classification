# Brain Tumor MRI Classification
# Veri seti görselleştirme dosyası

import os
import matplotlib.pyplot as plt
from PIL import Image

train_path = "veri_seti/Training"
test_path = "veri_seti/Testing"

os.makedirs("ciktilar", exist_ok=True)

class_names = sorted(os.listdir(train_path))

# =========================
# Sınıf Dağılımı Grafiği
# =========================

train_counts = []
test_counts = []

for class_name in class_names:
    train_class_path = os.path.join(train_path, class_name)
    test_class_path = os.path.join(test_path, class_name)

    train_counts.append(len(os.listdir(train_class_path)))
    test_counts.append(len(os.listdir(test_class_path)))

plt.figure(figsize=(9, 5))
x = range(len(class_names))

plt.bar(x, train_counts, width=0.4, label="Eğitim")
plt.bar([i + 0.4 for i in x], test_counts, width=0.4, label="Test")

plt.xticks([i + 0.2 for i in x], class_names)
plt.xlabel("Sınıflar")
plt.ylabel("Görüntü Sayısı")
plt.title("Veri Seti Sınıf Dağılımı")
plt.legend()
plt.tight_layout()
plt.savefig("ciktilar/veri_dagilimi.png")
plt.close()

# =========================
# Örnek MRI Görüntüleri
# =========================

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    class_path = os.path.join(train_path, class_name)
    image_name = os.listdir(class_path)[0]
    image_path = os.path.join(class_path, image_name)

    image = Image.open(image_path)

    plt.subplot(2, 2, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(class_name)
    plt.axis("off")

plt.suptitle("Her Sınıftan Örnek MRI Görüntüleri")
plt.tight_layout()
plt.savefig("ciktilar/ornek_mri_gorselleri.png")
plt.close()

# =========================
# Model Karşılaştırma Grafiği
# =========================

model_names = ["Temel CNN", "MobileNetV2"]
accuracies = [84.75, 77.44]

plt.figure(figsize=(7, 5))
plt.bar(model_names, accuracies)
plt.ylim(0, 100)
plt.ylabel("Test Accuracy (%)")
plt.title("Model Başarı Karşılaştırması")

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"%{value}", ha="center")

plt.tight_layout()
plt.savefig("ciktilar/model_karsilastirma.png")
plt.close()

print("Görseller başarıyla oluşturuldu.")
print("Oluşan dosyalar:")
print("- ciktilar/veri_dagilimi.png")
print("- ciktilar/ornek_mri_gorselleri.png")
print("- ciktilar/model_karsilastirma.png")