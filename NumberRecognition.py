import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Pobranie zestawu danych MNIST - zbiór obrazów ręcznie pisanych cyfr (0-9)
# Dane są automatycznie pobierane i transformowane do tensora PyTorch
training_data = datasets.MNIST(
    root="data",  # Katalog, w którym będą przechowywane dane
    train=True,  # Pobieranie danych treningowych
    download=True,  # Pobranie danych, jeśli nie są jeszcze dostępne
    transform=ToTensor(),  # Konwersja obrazów do tensorów PyTorch
)

test_data = datasets.MNIST(
    root="data",
    train=False,  # Pobieranie danych testowych
    download=True,
    transform=ToTensor(),
)

# Ustalenie rozmiaru wsadu (batch size), czyli liczby obrazów przetwarzanych jednocześnie
batch_size = 64

# Tworzenie DataLoaderów do efektywnego ładowania danych w porcjach (batch loading)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # Losowe mieszanie danych w każdej epoce

test_dataloader = DataLoader(test_data, batch_size=batch_size)  # Dane testowe nie muszą być mieszane

# Sprawdzenie wymiarów pojedynczej partii danych
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # N: liczba obrazów w wsadzie, C: liczba kanałów (1 - skala szarości), HxW: wymiary obrazu
    print(f"Shape of y: {y.shape} {y.dtype}")  # Wektory klas
    break

# Sprawdzenie dostępności GPU i wybór urządzenia
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Definicja modelu sieci neuronowej
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Spłaszczenie obrazu 28x28 do jednowymiarowego wektora 784-elementowego
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # Warstwa w pełni połączona (784 wejścia -> 512 neuronów)
            nn.BatchNorm1d(512),  # Batch normalization (normalizacja wsadowa)
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.Dropout(0.2),  # Dropout (losowe wyłączanie 20% neuronów)

            nn.Linear(512, 512),  # Kolejna warstwa w pełni połączona
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.Dropout(0.2),  # Dropout

            nn.Linear(512, 10)  # Ostateczna warstwa (10 neuronów dla każdej cyfry 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # Przekazanie przez kolejne warstwy
        return logits

# Inicjalizacja modelu i przeniesienie go na GPU (jeśli dostępne)
model = NeuralNetwork().to(device)
print(model)

# Definicja funkcji kosztu i optymalizatora
loss_fn = nn.CrossEntropyLoss()  # Funkcja strat dla klasyfikacji wieloklasowej
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optymalizator Adam z domyślnym learning rate = 0.001

# Listy do śledzenia strat i dokładności w trakcie treningu
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Funkcja treningowa
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Liczba próbek w zbiorze treningowym
    model.train()  # Ustawienie modelu w tryb treningowy

    total_loss, correct = 0, 0  # Inicjalizacja zmiennych do śledzenia wyników

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Przeniesienie danych na odpowiednie urządzenie

        pred = model(X)  # Przewidywanie
        loss = loss_fn(pred, y)  # Obliczenie straty

        loss.backward()  # Propagacja wsteczna (obliczanie gradientów)
        optimizer.step()  # Aktualizacja wag
        optimizer.zero_grad()  # Zerowanie gradientów

        total_loss += loss.item() #pobiera wartość numeryczną straty, aby można ją było dodać do całkowitej straty w danej epoce.
        correct += (pred.argmax(1) == y).type(torch.float).sum().item() # wybiera najbardziej prawdopodobną klasę dla każdej próbki, a następnie sprawdza, ile z nich zgadza się z rzeczywistymi etykietami y, sumując poprawne predykcje.

        # Co 100 batchy wyświetlamy aktualny stan procesu uczenia
        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    train_losses.append(total_loss / len(dataloader))  # Średnia strata w epoce
    train_accuracies.append(correct / size)  # Dokładność

# Funkcja testująca model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    test_loss, correct = 0, 0

    with torch.no_grad():  # Wyłączamy obliczanie gradientów dla oszczędności pamięci
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_losses.append(test_loss / num_batches)
    test_accuracies.append(correct / size)

    print(f"Test Error: \n Accuracy: {(100*(correct/size)):>0.1f}%, Avg loss: {test_loss / num_batches:>8f} \n")

# Trenowanie modelu
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Zapisanie modelu do pliku
torch.save(model.state_dict(), "mnist_model.pth")
print("Saved PyTorch Model State to mnist_model.pth")

# Wizualizacja wyników
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
epochs_range = range(epochs)

axs[0].plot(epochs_range, train_losses, label="Training", color="blue")
axs[0].plot(epochs_range, test_losses, label="Test", color="green")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs_range, train_accuracies, label="Training", color="blue")
axs[1].plot(epochs_range, test_accuracies, label="Test", color="green")
axs[1].set_xlabel("# Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].grid(True)

plt.show()

# Wizualizacja predykcji dla 9 losowych obrazów
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
model.eval()
for ax in axes.flat:
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        predicted_label = pred.argmax(1).item()
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Pred: {predicted_label}, True: {label}")
    ax.axis("off")

plt.show()

