import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import onnx
from onnx2pytorch import ConvertModel
from head.metrics import ArcFace  # Ensure this file exists in your project structure
from PIL import Image
import numpy as np

# ===============================================
# 1. Configuration
# ===============================================
config = {
    'DATA_ROOT': r'C:\SKOLI\HELGI\LV\datasets\training\my_face_dataset',  # Update to your dataset root
    'MODEL_SAVE_PATH': r'C:\SKOLI\HELGI\LV\finetuned_model.pth',
    'ONNX_SAVE_PATH': r'C:\SKOLI\HELGI\LV\finetuned_model.onnx',  # Path for ONNX export
    'INPUT_SIZE': (112, 112),  # Common face crop size
    'NUM_CLASSES': 2,          # "my face" and "others"
    'LR': 1e-4,
    'BATCH_SIZE': 1,           # Must be 1 because of ONNX conversion limitation
    'NUM_EPOCHS': 10,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # Path to the ONNX file downloaded by InsightFace for buffalo_sc recognition.
    'ONNX_MODEL_PATH': r'C:\Users\olafu\.insightface\models\buffalo_sc\w600k_mbf.onnx',
    # Reference image path (should be a clear image of you)
    'REFERENCE_IMAGE_PATH': r"C:\SKOLI\HELGI\LV\datasets\testing\reference\train_14-15-58-49.jpg"
}

# ===============================================
# 2. Data Preparation (for Training)
# ===============================================
# Expected dataset structure:
#   my_face_dataset/
#       my_face/    (images of you)
#       others/     (images of others)
transform = transforms.Compose([
    transforms.Resize(config['INPUT_SIZE']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
dataset = datasets.ImageFolder(config['DATA_ROOT'], transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)

# ===============================================
# 3. Convert ONNX Pretrained Weights to PyTorch (Backbone)
# ===============================================
print("Loading ONNX model from:", config['ONNX_MODEL_PATH'])
onnx_model = onnx.load(config['ONNX_MODEL_PATH'])
backbone = ConvertModel(onnx_model)
backbone.train()
print("Converted ONNX model to PyTorch. Model structure:")
print(backbone)

# Freeze backbone parameters so that they are not updated during fine-tuning.
for param in backbone.parameters():
    param.requires_grad = False

# ===============================================
# 4. Attach a New ArcFace Head
# ===============================================
if hasattr(backbone, 'last_linear'):
    in_features = backbone.last_linear.in_features
else:
    in_features = 512  # fallback value if not found
    print("Warning: Using default in_features =", in_features)

head = ArcFace(in_features=in_features, out_features=config['NUM_CLASSES'], device_id=[0])
for param in head.parameters():
    param.requires_grad = True

# ===============================================
# 5. Combine Backbone and Head into One Model
# ===============================================
class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone, head):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, label=None):
        # Run backbone in no_grad mode to use fixed BatchNorm statistics.
        with torch.no_grad():
            embedding = self.backbone(x)
        if label is not None:
            logits = self.head(embedding, label)
            return logits
        return embedding

    def train(self, mode=True):
        # Override train() to keep backbone in eval mode and head in train mode.
        self.backbone.eval()
        self.head.train(mode)
        return self

model = FaceRecognitionModel(backbone, head)
model = model.to(config['DEVICE'])

# ===============================================
# 6. Define Loss and Optimizer
# ===============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=config['LR'])  # Only update head parameters

# ===============================================
# 7. Training Loop
# ===============================================
model.train()  # Ensures backbone remains in eval mode while head is training.
for epoch in range(config['NUM_EPOCHS']):
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(config['DEVICE'])
        labels = labels.to(config['DEVICE'])
        
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}], Loss: {avg_loss:.4f}")

# ===============================================
# 8. Save the Fine-Tuned Model (PyTorch)
# ===============================================
torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])
print("Model saved to:", config['MODEL_SAVE_PATH'])

# ===============================================
# 9. Export the Fine-Tuned Model to ONNX for C Inference
# ===============================================
# Create an inference wrapper that calls model(x, label=None)
class InferenceModel(nn.Module):
    def __init__(self, model):
        super(InferenceModel, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x, label=None)

inference_model = InferenceModel(model)
inference_model.eval()

dummy_input = torch.randn(1, 3, config['INPUT_SIZE'][0], config['INPUT_SIZE'][1], device=config['DEVICE'])
torch.onnx.export(
    inference_model,
    dummy_input,
    config['ONNX_SAVE_PATH'],
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    opset_version=11
)
print("Model exported to ONNX format as:", config['ONNX_SAVE_PATH'])

# ===============================================
# 10. Generate and Save Reference Embedding
# ===============================================
# This computes the embedding from a known reference image of your face.
ref_img = Image.open(config['REFERENCE_IMAGE_PATH']).convert("RGB")
ref_tensor = transform(ref_img).unsqueeze(0).to(config['DEVICE'])
with torch.no_grad():
    ref_embedding = inference_model(ref_tensor)
ref_embedding_np = ref_embedding.cpu().numpy().flatten()
ref_embedding_path = os.path.join(os.path.dirname(config['MODEL_SAVE_PATH']), "ref_embedding.txt")
np.savetxt(ref_embedding_path, ref_embedding_np)
print("Reference embedding saved to:", ref_embedding_path)
