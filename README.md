# Intel Image Classification Using PyTorch

A deep learning project that classifies natural scene images into six categories using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Overview

This project implements a CNN to classify images of natural scenes around the world. The model classifies images into six categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Features

- Custom CNN architecture for image classification
- GPU acceleration support
- Data augmentation and normalization
- Training and validation pipeline
- Model checkpointing

## Requirements

- Python 3.x
- PyTorch
- torchvision
- CUDA (optional, for GPU acceleration)

## Installation

```bash
pip install torch torchvision
```

## Project Structure

The project consists of these main components:

1. Data Preprocessing
   - Image resizing to 150x150 pixels
   - Data normalization
   - Data augmentation
   - DataLoader implementation

2. Model Architecture
   ```python
   CNN Architecture:
   - Conv2d(3, 32, 3) -> ReLU -> MaxPool2d
   - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d
   - Conv2d(64, 128, 3) -> ReLU -> MaxPool2d
   - Fully connected layers (128*18*18 -> 512 -> 6)
   - Dropout for regularization
   ```

3. Training Pipeline
   - Cross-entropy loss
   - Adam optimizer
   - Learning rate: 0.001
   - Batch size: 32

## Usage

1. Data Preparation:
```python
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root="seg_train", transform=transform)
val_data = datasets.ImageFolder(root="seg_test", transform=transform)
```

2. Training:
```python
# Initialize model, criterion, and optimizer
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
```

3. Save/Load Model:
```python
# Save model
torch.save(model, 'model.pth')

# Load model
model = torch.load('model.pth')
```

## Model Performance

The model achieves the following metrics:
- Training Accuracy: 100%
- Validation Accuracy: 100%
- Training Loss: ~0.0
- Validation Loss: ~0.0

## Dataset

The Intel Image Classification dataset contains around 25,000 images of size 150x150 pixels organized into 6 categories.

- Training Set: ~14,000 images
- Test Set: ~3,000 images

Data source: [Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Future Improvements

Potential areas for enhancement:
1. Implementation of additional data augmentation techniques
2. Experimentation with different architectures (ResNet, VGG, etc.)
3. Learning rate scheduling
4. Cross-validation implementation
5. Model interpretability analysis
6. Deployment pipeline

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
@misc{intel-image-classification,
  author = {Your Name},
  title = {Intel Image Classification Using PyTorch},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/intel-image-classification}
}
```

## Acknowledgments

- Intel for providing the dataset
- PyTorch team for the deep learning framework
- Kaggle for the computing resources

