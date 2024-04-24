from LeNet import LeNet
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import train_dataset, test_dataset, plot_image

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


def train(epoch: int, stream):
    lenet.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = lenet(images)
        loss = criterion(output, labels)
        if i % 500 == 0:
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.detach().item()}')
            stream.write(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.detach().item()}\n')
        loss_list.append(loss.detach().item())
        batch_list.append(i)
        loss.backward()
        optimizer.step()


def test(stream):
    lenet.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_dataloader):
        output = lenet(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(test_dataset)
    print(f'Accuracy: {total_correct}/{len(test_dataset)} ({total_correct * 100 / len(test_dataset):.0f}%)')
    stream.write(f'Accuracy: {total_correct}/{len(test_dataset)} ({total_correct * 100 / len(test_dataset):.0f}%)\n')


if __name__ == "__main__":
    lenet = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet.parameters(), lr=2e-3)
    fw = open("loss.log", "w", encoding="utf-8")
    for epoch in range(5):
        train(epoch, fw)
        test(fw)
    fw.close()
    # save model
    torch.save(lenet.state_dict(), 'lenet.pth')
    plot_image(test_dataset[100][0])
    print(lenet(test_dataset[100][0].unsqueeze(0)).argmax())
