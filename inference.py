# To load our best saved model, we can use "load_state_dict"
model_cnn = FlowerNet()
model_cnn.to(device)
model_cnn.load_state_dict(torch.load('Flower_CNN.pth',map_location=torch.device('cuda')))