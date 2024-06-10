import torch
from torchvision import transforms
from PIL import Image
from utils.segment import split_image
from mnist_classify import load_model

def predict(model, device, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred = output.max(1, keepdim=True)[1]
    return pred.item()

def predict_on_segments(segment_paths, model, device):
    results = []
    for segment_path in segment_paths:
        result = predict(model, device, segment_path)
        results.append(result)
    return results

def predictions_to_string(predictions):
    return ''.join(map(str, predictions))

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './models/mnist_cnn.pt'
    model = load_model(model_path, DEVICE)
    
    # 需要预测的文件的路径
    image_path='result.png'
    segment_image_paths = split_image(image_path, 32, 32)
    
    # Predict on each segment
    predictions = predict_on_segments(segment_image_paths, model, DEVICE)
    
    # Convert predictions to string
    predictions_str = predictions_to_string(predictions)
    
    # Print combined predictions as string
    print(f"Predictions: {predictions_str}")
