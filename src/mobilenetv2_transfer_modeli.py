# Brain Tumor MRI Classification

import os  # klasör işlemleri
import numpy as np  # sayısal işlemler
import tensorflow as tf  # yapay zeka işlemleri
import matplotlib.pyplot as plt  # grafik çizmek için
import seaborn as sns  # confusion matrix için

from tensorflow.keras import layers, models  # model katmanları
from tensorflow.keras.applications import MobileNetV2  # hazır model
from sklearn.metrics import confusion_matrix, classification_report  # performans ölçümleri

train_path = "veri_seti/Training"  # eğitim verileri
test_path = "veri_seti/Testing"  # test verileri

IMG_SIZE = 224  # resim boyutu
BATCH_SIZE = 32  # batch sayısı
EPOCHS = 5  # eğitim tekrar sayısı

os.makedirs("ciktilar", exist_ok=True)  # çıktı klasörü oluşturur
os.makedirs("modeller", exist_ok=True)  # model klasörü oluşturur

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,  # eğitim klasörü
    image_size=(IMG_SIZE, IMG_SIZE),  # resimleri boyutlandırır
    batch_size=BATCH_SIZE,  # batch ayarı
    shuffle=True  # verileri karıştırır
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,  # test klasörü
    image_size=(IMG_SIZE, IMG_SIZE),  # resimleri boyutlandırır
    batch_size=BATCH_SIZE,  # batch ayarı
    shuffle=False  # test verisini karıştırmaz
)

class_names = train_dataset.class_names  # sınıf isimlerini alır
print("Sınıflar:", class_names)  # sınıfları yazdırır

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),  # resmi çevirir
    layers.RandomRotation(0.1),  # resmi döndürür
    layers.RandomZoom(0.1),  # yakınlaştırma yapar
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input  # mobilenet ön işleme

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)  # eğitim verisini hazırlar
)

test_dataset = test_dataset.map(
    lambda x, y: (preprocess_input(x), y)  # test verisini hazırlar
)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # eğitimi hızlandırır
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # testi hızlandırır

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),  # giriş boyutu
    include_top=False,  # son katmanı kaldırır
    weights="imagenet"  # hazır ağırlıkları kullanır
)

base_model.trainable = False  # temel katmanları dondurur

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # giriş katmanı

    data_augmentation,  # veri artırma

    base_model,  # hazır model

    layers.GlobalAveragePooling2D(),  # boyutu küçültür

    layers.Dense(128, activation="relu"),  # öğrenme katmanı
    layers.Dropout(0.5),  # ezberi azaltır

    layers.Dense(4, activation="softmax")  # 4 sınıflı çıktı verir
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # optimizer ayarı
    loss="sparse_categorical_crossentropy",  # loss hesabı
    metrics=["accuracy"]  # başarı metriği
)

model.summary()  # model özetini gösterir

history = model.fit(
    train_dataset,  # eğitim verisi
    validation_data=test_dataset,  # doğrulama verisi
    epochs=EPOCHS  # epoch sayısı
)

test_loss, test_accuracy = model.evaluate(test_dataset)  # modeli test eder

print("\nMobileNetV2 Test Loss:", test_loss) 
print("MobileNetV2 Test Accuracy:", test_accuracy)  

plt.figure(figsize=(8, 5))  # grafik boyutu
plt.plot(history.history["accuracy"], label="eğitim accuracy")  # eğitim başarısı
plt.plot(history.history["val_accuracy"], label="doğrulama accuracy")  # doğrulama başarısı
plt.title("mobilenetv2 accuracy grafiği")  # grafik başlığı
plt.xlabel("epoch")  # x ekseni
plt.ylabel("accuracy")  # y ekseni
plt.legend()  # açıklama kutusu
plt.savefig("ciktilar/mobilenetv2_accuracy_grafigi.png")  # kaydeder
plt.close()  # kapatır

plt.figure(figsize=(8, 5))  # grafik boyutu
plt.plot(history.history["loss"], label="eğitim loss")  # eğitim kaybı
plt.plot(history.history["val_loss"], label="doğrulama loss")  # doğrulama kaybı
plt.title("mobilenetv2 loss grafiği")  # grafik başlığı
plt.xlabel("epoch")  # x ekseni
plt.ylabel("loss")  # y ekseni
plt.legend()  # açıklama kutusu
plt.savefig("ciktilar/mobilenetv2_loss_grafigi.png")  # kaydeder
plt.close()  # kapatır

y_true = []  # gerçek etiketler
y_pred = []  # tahmin edilen etiketler

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)  # tahmin yapar
    predicted_classes = np.argmax(predictions, axis=1)  # en yüksek olasılığı seçer

    y_true.extend(labels.numpy())  # gerçek etiketleri ekler
    y_pred.extend(predicted_classes)  # tahminleri ekler

cm = confusion_matrix(y_true, y_pred)  # confusion matrix oluşturur

plt.figure(figsize=(8, 6))  # grafik boyutu

sns.heatmap(
    cm,
    annot=True,  # sayı gösterir
    fmt="d",  # tam sayı formatı
    cmap="Blues",  # renk ayarı
    xticklabels=class_names,  # x etiketleri
    yticklabels=class_names  # y etiketleri
)

plt.title("mobilenetv2 confusion matrix")  # başlık
plt.xlabel("tahmin edilen sınıf")  # x ekseni
plt.ylabel("gerçek sınıf")  # y ekseni
plt.savefig("ciktilar/mobilenetv2_confusion_matrix.png")  # kaydeder
plt.close()  # kapatır

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nmobilenetv2 classification report:\n")  
print(report)  

with open("ciktilar/mobilenetv2_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)  # txt dosyasına kaydeder

model.save("modeller/mobilenetv2_beyin_tumoru_modeli.keras")  # modeli kaydeder

print("\nmobilenetv2 modeli ve çıktılar kaydedildi")  