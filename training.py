import torch

def train(model, epochs, optimizer, loss, loader):

    outputs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for data, _ in loader:
            data = data.view(data.shape[0], model.layer_dims[0]).to(device)
            
            reconstructed = model(data)
            loss = loss(reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        outputs.append((epoch, data, reconstructed))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return outputs, losses, model