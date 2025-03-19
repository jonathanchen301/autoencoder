import torch
import visualization

def train(model, epochs, optimizer, loss_function, loader):

    outputs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        for batch in loader:
            data = batch.to(device)
            
            reconstructed = model(data)
            loss = loss_function(reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)

        outputs.append((epoch, data.cpu().detach(), reconstructed.cpu().detach()))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    visualization.plot_losses(losses, range(epochs))
    visualization.reconstructed_vs_data(loader, model, 1)

    return outputs, losses, model