import torch
import os.path

MODEL_FOLDER_PATH = os.path.join('networks', 'saved_models')

def save_model(model: torch.nn.Module, file_name='model.pt') -> str:
    save_path = os.path.join(MODEL_FOLDER_PATH, file_name)
    with open(save_path, 'wb') as f:
        torch.save(model, f)
        return save_path

def load_model(file_name='model.pt'):
    with open(os.path.join(MODEL_FOLDER_PATH, file_name), 'rb') as f:
        return torch.load(f)

