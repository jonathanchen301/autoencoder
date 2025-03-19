import matplotlib.pyplot as plt

def plot_losses(losses, epochs):

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def reconstructed_vs_data(dataloader, model, dims):

    if dims == 1:

        model.eval()
        dataiter = iter(dataloader)
        row = next(dataiter)

        reconstructed = model(row)

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 3))
        for i in range(5):
            axes[0, i].plot(row[i])
            axes[0, i].axis('off')
            axes[1, i].plot(reconstructed[i].detach())
            axes[1, i].axis('off')

        plt.show()