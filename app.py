import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define function to get features from the VGG model
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_3', '26': 'relu4_3'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# Define function to calculate gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Define function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image

# Define function to convert a tensor image to a PIL image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Define function to perform style transfer
# Define function to perform style transfer
def stylize_image(content_path, style_path, num_steps=500, style_weight=1000000, content_weight=1):
    device = torch.device("cuda")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    # Get content and style features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Initialize the target image as a copy of the content image
    target = content.clone().requires_grad_(True).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam([target], lr=0.01)

    for step in range(num_steps):
        target_features = get_features(target, vgg)

        # Calculate content loss
        content_loss = torch.mean((content_features['relu3_3'] - target_features['relu3_3']) ** 2)

        # Calculate style loss
        style_loss = 0
        for layer in style_features:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_features[layer])
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (target_feature.size(1) * target_feature.size(2) * target_feature.size(3))

        # Calculate total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Update the target image
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)  # Add retain_graph=True here
        optimizer.step()

        # Print progress
        if (step + 1) % 100 == 0:
            st.write(f"Step [{step+1}/{num_steps}], Loss: {total_loss.item()}")

    # Convert the target image tensor to a PIL image
    stylized_image = tensor_to_image(target)

    return stylized_image

# Main function
def main():
    st.title("Neural Style Transfer")
    content_image_path = st.file_uploader("Upload Content Image", type=["png", "jpg"])
    style_image_path = st.file_uploader("Upload Style Image", type=["png", "jpg"])

    if content_image_path and style_image_path:
        content_image = Image.open(content_image_path)
        style_image = Image.open(style_image_path)

        st.write("Content Image:")
        st.image(content_image)

        st.write("Style Image:")
        st.image(style_image)

        if st.button("Stylize"):
            stylized_image = stylize_image(content_image_path, style_image_path)
            st.write("Stylized Image:")
            st.image(stylized_image)

# Run the app
if __name__ == "__main__":
    main()
