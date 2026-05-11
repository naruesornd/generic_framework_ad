import torch
from model.lstm_model.enhanced_lstm import EnhancedLSTM

def load_model(model_path, X_train, device):
    print(X_train.shape[1])
    model = EnhancedLSTM(input_dim=X_train.shape[1], hidden_dim=128, output_dim=1, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

