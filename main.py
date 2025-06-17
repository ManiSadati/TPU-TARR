import os
import gzip
import pickle
import random
import copy

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from models.CustomConv import replace_conv_layers, CustomConv2d

# Set random seeds for reproducibility
random.seed(12345)
torch.manual_seed(12345)

def preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ImageNetDataset(Dataset):
    def __init__(self, image_paths, labels, root_dir, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_labels(val_maps_file):
    val_maps_file = os.path.join(os.path.dirname(__file__), val_maps_file)
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    map_label = {dirs[i]: i for i in range(1000)}
    labels = [map_label[m[1]] for m in mappings]
    idirs = [m[0] for m in mappings]
    print(idirs[:10])
    return idirs, labels

def load_models():
    models_dict = {
        "squeezenet1_0": torchvision.models.squeezenet1_0(pretrained=True).eval(),
        # Add more models as needed
    }
    return models_dict

def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted

def evaluate_models(data_loader, device, models_dict, batch_limit=None):
    total_correct = {name: 0 for name in models_dict}
    total = 0
    cnt_b = 0
    for images, true_labels in data_loader:
        images, true_labels = images.to(device), true_labels.to(device)
        total += true_labels.size(0)
        for name, model in models_dict.items():
            predicted_labels = predict(model, images)
            correct = (predicted_labels == true_labels).sum().item()
            total_correct[name] += correct
        if batch_limit is not None and cnt_b >= batch_limit - 1:
            break
        cnt_b += 1
        print("batch", cnt_b)
    accuracies = {name: (total_correct[name] / total * 100) for name in models_dict}
    for name, accuracy in accuracies.items():
        print(f"{name} - Total Accuracy: {accuracy:.2f}%")
    return accuracies

def plot_max_activations(max_tile_activations):
    for model_name, layers in max_tile_activations.items():
        num_layers = len(layers)
        num_rows = (num_layers + 4) // 5
        num_cols = min(5, num_layers)
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 3))
        fig.suptitle(f'Maximum Activations for {model_name}', fontsize=16)
        axs = axs.flatten() if num_layers > 1 else [axs]
        z = 0
        for ax, (layer, max_values) in zip(axs, layers.items()):
            flattened_values = [item for sublist in max_values for item in sublist]
            ax.hist(flattened_values, bins=30, color='blue', alpha=0.7)
            ax.set_title(f'Layer: {z}')
            ax.set_xlabel('Max Activation Value')
            ax.set_ylabel('Frequency')
            z += 1
        for i in range(num_layers, len(axs)):
            axs[i].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def clone_models(models_dict):
    return {name: copy.deepcopy(model) for name, model in models_dict.items()}

def hook_fn(module, input, output):
    current_max_values = torch.amax(output, dim=(0, 2, 3))
    if not hasattr(module, 'max_output'):
        module.max_output = current_max_values
    else:
        module.max_output = torch.max(module.max_output, current_max_values)

def attach_hooks(models_dict):
    for model in models_dict.values():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(hook_fn)

def get_max_values_and_evaluate(data_loader, device, models_dict, batch_limit=5):
    cloned_models = clone_models(models_dict)
    for model in cloned_models.values():
        replace_conv_layers(model)
    evaluate_models(data_loader, device, cloned_models, batch_limit=batch_limit)
    max_tile_activations = {}
    max_activations = {}
    for name, model in cloned_models.items():
        max_activations[name] = {}
        max_tile_activations[name] = {}
        print(f"Maximum activations for {name}:")
        l_counter = 0
        for layer in model.modules():
            if isinstance(layer, CustomConv2d) and hasattr(layer, 'ps'):
                if l_counter not in max_activations[name]:
                    max_activations[name][l_counter] = layer.max_layer
                    max_tile_activations[name][l_counter] = layer.ps
                    print(layer.ps)
                l_counter += 1
    return max_tile_activations, max_activations

def bit_flip(x):
    bits = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)
    bit_pos = random.randint(0, 31)
    bits[0] ^= 1 << bit_pos
    return np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

def inject_fault_hook(module, input, output):
    with torch.no_grad():
        for channel in range(output.shape[1]):
            h, w = random.randint(0, output.shape[2] - 1), random.randint(0, output.shape[3] - 1)
            output[0, channel, h, w] = bit_flip(output[0, channel, h, w].item())

def attach_fault_injection_hooks(cloned_models):
    for model in cloned_models.values():
        for layer in model.modules():
            if isinstance(layer, CustomConv2d):
                if hasattr(layer, 'InjectFault'):
                    layer.InjectFault = True
                else:
                    print("WTF")

def load_max_activations(models_dict):
    cloned_models = clone_models(models_dict)
    for model in cloned_models.values():
        replace_conv_layers(model)
    with open('checkpoint/max_activations.pkl', 'rb') as file:
        max_activations = pickle.load(file)
    with open('checkpoint/max_tile_activations.pkl', 'rb') as file:
        max_tile_activations = pickle.load(file)
    for name, model in cloned_models.items():
        l_counter = 0
        z = 0
        for layer in model.modules():
            if isinstance(layer, CustomConv2d):
                layer.max_layer = max_activations[name][l_counter]
                layer.ps = max_tile_activations[name][l_counter]
                layer.do_tile = True
                l_counter += 1
        print("yes", name, l_counter, z)
    return cloned_models

def main():
    dataset_path = '/home/mani/Downloads/ILSVRC2012_img_val/' # this should be the path to the ImageNet validation dataset
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('checkpoint/max_activations.pkl'):
        print("max_activations.pkl not found, running evaluation to generate it.")
        with open('checkpoint/max_activations.pkl', 'wb') as file:
            pickle.dump({}, file)


            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)
    val_maps_file = 'imagenet_val_maps.pklz'
    dirs, labels = get_labels(val_maps_file)
    transform = preprocess_transform()
    dataset = ImageNetDataset(dirs, labels, dataset_path, transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    models_dict = load_models()
    for model in models_dict.values():
        model.to(device)
    evaluate_models(data_loader, device, models_dict, batch_limit=1)
    print("---- get max values ------")
    max_tile_activations, max_activations = get_max_values_and_evaluate(data_loader, device, models_dict, batch_limit=100)
    with open('checkpoint/max_activations.pkl', 'wb') as file:
        pickle.dump(max_activations, file)
    with open('checkpoint/max_tile_activations.pkl', 'wb') as file:
        pickle.dump(max_tile_activations, file)
    plot_max_activations(max_tile_activations)
    cloned_models = load_max_activations(models_dict)
    attach_fault_injection_hooks(cloned_models)
    print("results of fault injection")
    evaluate_models(data_loader, device, cloned_models, batch_limit=100)
    for name, model in cloned_models.items():
        print("model", name)
        l_counter = 0
        for layer in model.modules():
            if isinstance(layer, CustomConv2d):
                print(f"layer {l_counter}:", layer.layer_detect, layer.tile_detect, layer.total_fault, layer.max_layer)
                l_counter += 1

if __name__ == "__main__":
    main()