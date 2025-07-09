import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.fftpack import dct, idct
from PIL import Image
import os
import random  


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

new_size = 224
mask = np.zeros((new_size, new_size), dtype=bool)
mask[::2, ::2] = True  
mask[1::2, 1::2] = True


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# DCT
def compute_dct_features(image, num_coeff=64):
    hann_window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    windowed_img = image * hann_window
    dct_result = dct(dct(windowed_img, axis=0, norm='ortho'), axis=1, norm='ortho')
    magnitude = np.log1p(np.abs(dct_result))
    coeff_mask = np.zeros_like(magnitude)
    coeff_mask[:num_coeff, :num_coeff] = 1
    magnitude = magnitude * coeff_mask
    phase = np.angle(dct_result.astype(np.complex128))
    return np.stack([magnitude, phase], axis=-1)



def process_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read image at {img_path}")
        return None

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
    aset_img = np.zeros_like(gray_img)
    aset_img[mask] = gray_img[mask]
    spatial_input = aset_img.copy()

    try:
        spatial_tensor = base_transform(spatial_input).unsqueeze(0).float()
    except Exception as e:
        print(f"Error in transforming spatial input: {e}")
        return None


    mid_row = aset_img[target_size[0] // 2, :]
    temporal_img = np.tile(mid_row, (target_size[0], 1))
    temporal_tensor = base_transform(temporal_img).unsqueeze(0).float()


    frequency_img = compute_dct_features(aset_img)
    frequency_tensor = torch.tensor(frequency_img).permute(2, 0, 1).unsqueeze(0).float()

  
    target_tensor = base_transform(gray_img).unsqueeze(0).float()

    return {
        'spatial': spatial_tensor,
        'temporal': temporal_tensor,
        'frequency': frequency_tensor,
        'target': target_tensor,
        'aset_img': aset_img,
        'full_img': gray_img
    }



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out



class EnhancedViTModel(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=768, heads=12, mlp_dim=3072, num_layers=12):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.spatial_chan = 64
        self.dim = dim

        self.spatial_preproc = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, self.spatial_chan, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.spatial_chan)
        )


        self.spatial_embed = nn.Linear(self.patch_size ** 2 * self.spatial_chan, dim)
        self.temporal_embed = nn.Linear(self.patch_size ** 2, dim)
        self.frequency_embed = nn.Linear(2 * patch_size ** 2, dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, spatial, temporal, frequency):

        spatial = self.spatial_preproc(spatial)
        bs, c, h, w = spatial.shape
        spatial = spatial.view(bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        spatial = spatial.permute(0, 2, 4, 1, 3, 5).reshape(bs, -1, c * self.patch_size ** 2)
        spatial_embed = self.spatial_embed(spatial)


        bs, _, h, w = temporal.shape
        temporal = temporal.view(bs, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        temporal = temporal.permute(0, 1, 3, 2, 4).reshape(bs, -1, self.patch_size ** 2)
        temporal_embed = self.temporal_embed(temporal)


        bs, _, h, w = frequency.shape
        frequency = frequency.view(bs, 2, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        frequency = frequency.permute(0, 2, 4, 1, 3, 5).reshape(bs, -1, 2 * self.patch_size ** 2)
        frequency_embed = self.frequency_embed(frequency)


        combined_embed = spatial_embed + 0.5 * temporal_embed + 0.5 * frequency_embed
        combined_embed = torch.cat([self.cls_token.expand(bs, -1, -1), combined_embed], dim=1)
        combined_embed += self.pos_embed[:, :(self.num_patches + 1)]
        combined_embed = self.dropout(combined_embed)


        for layer in self.encoder:
            combined_embed = layer(combined_embed)
        transformer_out = self.ln(combined_embed)


        transformer_out = transformer_out[:, 1:].permute(0, 2, 1).view(
            bs, self.dim,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size
        )


        decoded = self.decoder(transformer_out)
        return decoded



def postprocess_output(pred, mask, aset_img):

    pred_np = pred.squeeze().cpu().numpy()

    while pred_np.ndim > 2:
        pred_np = pred_np.squeeze(0)


    if pred_np.ndim != 2:
        if pred_np.shape[0] == 1:  
            pred_np = pred_np[0]
        else:
            pred_np = pred_np.mean(axis=0)  

    pred_np = (pred_np + 1) * 127.5
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)

    if pred_np.shape != mask.shape:
        pred_np = cv2.resize(pred_np, (mask.shape[1], mask.shape[0]))


    full_img = np.zeros_like(aset_img, dtype=np.uint8)
    full_img[mask] = aset_img[mask]  


    bset_mask = ~mask
    full_img[bset_mask] = pred_np[bset_mask].ravel() 

    return full_img



def diamond_predictor_with_confidence(gray_img, mask):
    height, width = gray_img.shape
    predicted_img = np.copy(gray_img)
    confidence_map = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                neighbors = []
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width and mask[ni, nj]:
                        neighbors.append(gray_img[ni, nj])

                if neighbors:
                    predicted_img[i, j] = np.median(neighbors)
                    confidence = len(neighbors) / 4.0  
                    if len(neighbors) > 1:
                        std_dev = np.std(neighbors)
                        if std_dev > 0:
                            confidence *= min(1.0, 20.0 / std_dev)
                    confidence_map[i, j] = min(confidence, 1.0)
                else:
                    predicted_img[i, j] = 128
                    confidence_map[i, j] = 0.1

    return predicted_img, confidence_map



def vit_predictor_with_confidence(model, test_data, mask, aset_img):
    model.eval()
    with torch.no_grad():
        output = model(
            test_data['spatial'].to(device),
            test_data['temporal'].to(device),
            test_data['frequency'].to(device)
        )

    vit_recon = postprocess_output(output, mask, aset_img)
    confidence_map = np.zeros_like(vit_recon, dtype=np.float32)

    for i in range(1, vit_recon.shape[0] - 1):
        for j in range(1, vit_recon.shape[1] - 1):
            if not mask[i, j]:
                neighbors = vit_recon[i - 1:i + 2, j - 1:j + 2].flatten()
                avg_diff = np.mean(np.abs(vit_recon[i, j] - neighbors))
                confidence = np.exp(-avg_diff / 20.0)
                confidence_map[i, j] = confidence

    return vit_recon, confidence_map


def integrate_predictions(diamond_pred, diamond_conf, vit_pred, vit_conf, mask):
    integrated_img = np.zeros_like(diamond_pred, dtype=np.uint8)
    integrated_img[mask] = diamond_pred[mask]  # Aset点保持不变

    for i in range(diamond_pred.shape[0]):
        for j in range(diamond_pred.shape[1]):
            if not mask[i, j]:
                w_diamond = diamond_conf[i, j]
                w_vit = vit_conf[i, j]
                total_conf = w_diamond + w_vit
                if total_conf > 0:
                    integrated_value = (w_diamond * diamond_pred[i, j] + w_vit * vit_pred[i, j]) / total_conf
                    integrated_img[i, j] = np.clip(integrated_value, 0, 255)

    return integrated_img



def save_error_heatmap(original, prediction, title, filename):

    error = cv2.absdiff(original, prediction)


    plt.figure(figsize=(8, 6))
    plt.imshow(error, cmap='hot', vmin=0, vmax=100)  
    plt.colorbar(label='Absolute Error')
    plt.title(title)
    plt.axis('off')


    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Saved error heatmap to {filename}")



def visualize_results(original, aset_img, diamond_pred, vit_pred, integrated_pred):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(aset_img, cmap='gray')
    plt.title('Aset Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(diamond_pred, cmap='gray')
    plt.title('Diamond Prediction')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(vit_pred, cmap='gray')
    plt.title('ViT Prediction')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(integrated_pred, cmap='gray')
    plt.title('Integrated Prediction')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    diff = cv2.absdiff(original, integrated_pred)
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.title('Integrated Error')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    plt.show()

    def calculate_metrics(original, pred):
        mse = np.mean((original.astype(float) - pred.astype(float)) ** 2)
        psnr = 10 * np.log10((255 ** 2) / (mse + 1e-10))
        return mse, psnr

    mse_diamond, psnr_diamond = calculate_metrics(original, diamond_pred)
    mse_vit, psnr_vit = calculate_metrics(original, vit_pred)
    mse_integrated, psnr_integrated = calculate_metrics(original, integrated_pred)

    print("=" * 60)
    print(f"Diamond Predictor - MSE: {mse_diamond:.2f}, PSNR: {psnr_diamond:.2f} dB")
    print(f"ViT Predictor - MSE: {mse_vit:.2f}, PSNR: {psnr_vit:.2f} dB")
    print(f"Integrated Predictor - MSE: {mse_integrated:.2f}, PSNR: {psnr_integrated:.2f} dB")
    print("=" * 60)



def main(test_img_path='boat.png'):
   
    set_seed(42)

    if not os.path.exists(test_img_path):
        print(f"Error: Test image not found at {test_img_path}")
        return

    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Pre-trained model not found at {model_path}")
        print("Please make sure you have a trained model saved as 'best_model.pth'")
        return

    model = EnhancedViTModel(num_layers=12).to(device)

    try:
 
        state_dict = torch.load(model_path, map_location=device)

    
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}


        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict, strict=False)

        print("Successfully loaded compatible weights from pre-trained model")
        print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} parameters")


        missing_keys = [k for k in state_dict.keys() if k not in filtered_state_dict]
        if missing_keys:
            print("\nThe following weights could not be loaded due to size mismatch:")
            for k in missing_keys[:10]:  
                print(
                    f"- {k}: expected {model_state_dict[k].shape if k in model_state_dict else 'missing'}, got {state_dict[k].shape}")
            if len(missing_keys) > 10:
                print(f"... and {len(missing_keys) - 10} more mismatched weights")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()


    print("Processing test image...")
    test_data = process_image(test_img_path)
    if test_data is None:
        print("Error: Failed to process test image")
        return

    original_img = test_data['full_img']
    aset_img = test_data['aset_img']

    print("Running diamond predictor...")
    diamond_pred, diamond_conf = diamond_predictor_with_confidence(aset_img, mask)

    print("Running ViT predictor...")
    vit_pred, vit_conf = vit_predictor_with_confidence(model, test_data, mask, aset_img)

    print("Integrating predictions...")
    integrated_pred = integrate_predictions(diamond_pred, diamond_conf, vit_pred, vit_conf, mask)


    print("Generating and saving error heatmaps...")
    save_error_heatmap(original_img, diamond_pred,
                       'Diamond Predictor Error Distribution',
                       'diamond_predictor_error_heatmap.png')

    save_error_heatmap(original_img, vit_pred,
                       'ViT Predictor Error Distribution',
                       'vit_predictor_error_heatmap.png')

    save_error_heatmap(original_img, integrated_pred,
                       'Integrated Predictor Error Distribution',
                       'integrated_predictor_error_heatmap.png')


    print("Visualizing results...")
    visualize_results(original_img, aset_img, diamond_pred, vit_pred, integrated_pred)

    cv2.imwrite('diamond_prediction.png', diamond_pred)
    cv2.imwrite('vit_prediction.png', vit_pred)
    cv2.imwrite('integrated_prediction.png', integrated_pred)
    cv2.imwrite('original_image.png', original_img)
    cv2.imwrite('aset_image.png', aset_img)

    print("Done! All results saved to disk.")


if __name__ == "__main__":
    main()