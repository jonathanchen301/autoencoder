import matplotlib.pyplot as plt

def plot_losses(losses):

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()