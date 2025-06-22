import sys
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import streamlit as st

#Определение простой сверточной сети (тот же, что и в обучении)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        #Автоматический расчет размера
        with torch.no_grad():
            x = torch.zeros(1, 3, 100, 100)
            x = self.conv(x)
            n_size = x.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

#Сиамская сеть: 2 раза одна и та же CNN, выводы сравниваем
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()
    def forward(self, x1, x2):
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
        return out1, out2

#Функция для проверки похожести пары изображений
def is_same(model, img_path1, img_path2, transform, device, threshold=0.5):
    model.eval()
    img1 = transform(Image.open(img_path1).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(img_path2).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        out1, out2 = model(img1, img2)
        dist = nn.functional.pairwise_distance(out1, out2)
        print(f"Расстояние между изображениями: {dist.item():.4f}")
        return dist.item() < threshold, dist.item()

#Главная функция, которая загружает модель и сравнивает изображения
def main(img_path1, img_path2, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    #Создаем и загружаем обученную модель
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("siamese_model4.pth", map_location=device))
    same, distance = is_same(model, img_path1, img_path2, transform, device, threshold)
    print(f"Изображения похожи? {'Да' if same else 'Нет'}, расстояние={distance:.4f}")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(Image.open(img_path1))
    axs[0].set_title("Изображение 1")
    axs[0].axis('off')
    axs[1].imshow(Image.open(img_path2))
    axs[1].set_title("Изображение 2")
    axs[1].axis('off')
    st.write(f"Похожие: {'Да' if same else 'Нет'} (расстояние={distance:.4f})")
    plt.suptitle(f"Похожие: {'Да' if same else 'Нет'} (расстояние={distance:.4f})")
    plt.show()

def run_site():
    st.set_page_config(page_title="Сравнение фото", layout="centered")
    st.title("📷 Сравнение двух изображений")
    st.write("Загрузите два изображения, чтобы проверить, похожи ли они друг на друга.")

    uploaded_file1 = st.file_uploader("Выберите первое изображение", type=["jpg", "jpeg", "png"], key="img1")
    uploaded_file2 = st.file_uploader("Выберите второе изображение", type=["jpg", "jpeg", "png"], key="img2")

    threshold = st.slider("Порог расстояния для определения похожести", 0.0, 2.0, 0.5, step=0.01)

    if st.button("Сравнить"):
        if not uploaded_file1 or not uploaded_file2:
            st.warning("Пожалуйста, загрузите оба изображения.")
        else:
            with open("temp1.jpg", "wb") as f:
                f.write(uploaded_file1.getbuffer())
            with open("temp2.jpg", "wb") as f:
                f.write(uploaded_file2.getbuffer())

            main("temp1.jpg", "temp2.jpg", threshold)

            col1, col2 = st.columns(2)
            col1.image(uploaded_file1, caption="Изображение 1", use_container_width=True)
            col2.image(uploaded_file2, caption="Изображение 2", use_container_width=True)
    

if __name__ == "__main__":
    #if len(sys.argv) < 2 or sys.argv[1] != "gui":
        #if len(sys.argv) < 3:
        #    print("Использование: python compare.py <путь_к_изображению1> <путь_к_изображению2> [порог]")
        #    sys.exit(1)
        #img1 = sys.argv[1]
        #img2 = sys.argv[2]
       # thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        #if not (os.path.isfile(img1) and os.path.isfile(img2)):
        #    print("Ошибка: один или оба пути к файлам указаны неверно.")
       #     sys.exit(1)
       # main(img1, img2, thr)
    #else:
        # Работаем в режиме GUI (Streamlit)
        run_site()

