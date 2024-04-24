from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
train_dataset = MNIST(root='D:/temp/pytorch_cached_data', train=True, download=True, transform=trans)
test_dataset = MNIST(root='D:/temp/pytorch_cached_data', train=False, download=True, transform=trans)


def plot_image(img):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.show()


if __name__ == "__main__":
    # 查看数据集中的第一个元素
    print(train_dataset[0])
    # 查看数据集中的第一个元素的张量的最值
    print(train_dataset[0][0].min(), train_dataset[0][0].max())
    plot_image(train_dataset[0][0])
