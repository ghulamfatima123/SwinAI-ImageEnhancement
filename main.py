import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from models.network_swinir import SwinIR

def load_swinir_model(checkpoint_path, device):
    # Parse model type from checkpoint path
    if '001_classicalSR' in checkpoint_path:
        # Classical SR model
        if 'x2' in checkpoint_path:
            upscale = 2
        elif 'x3' in checkpoint_path:
            upscale = 3
        elif 'x4' in checkpoint_path:
            upscale = 4
        elif 'x8' in checkpoint_path:
            upscale = 8
        else:
            upscale = 4
            
        if 'SwinIR-M' in checkpoint_path:
            # Medium size model
            model = SwinIR(
                upscale=upscale,
                in_chans=3,
                img_size=48 if 'DIV2K' in checkpoint_path else 64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )
    elif '002_lightweight' in checkpoint_path:
        # Lightweight SR model
        if 'x2' in checkpoint_path:
            upscale = 2
        elif 'x3' in checkpoint_path:
            upscale = 3
        elif 'x4' in checkpoint_path:
            upscale = 4
        else:
            upscale = 4
            
        model = SwinIR(
            upscale=upscale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv'
        )
    elif '003_realSR' in checkpoint_path:
        # Real-world SR model
        if 'x2' in checkpoint_path:
            upscale = 2
        else:
            upscale = 4
            
        if 'SwinIR-L' in checkpoint_path:
            # Large model
            model = SwinIR(
                upscale=upscale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv'
            )
        else:
            # Medium model (SwinIR-M)
            model = SwinIR(
                upscale=upscale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv'
            )
    elif '004_grayDN' in checkpoint_path or '005_colorDN' in checkpoint_path:
        # Denoising model
        model = SwinIR(
            upscale=1,
            in_chans=3 if '005_colorDN' in checkpoint_path else 1,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
    elif '006_CAR' in checkpoint_path:
        # Compression artifact reduction model
        is_color = 'colorCAR' in checkpoint_path
        model = SwinIR(
            upscale=1,
            in_chans=3 if is_color else 1,
            img_size=126,
            window_size=7,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv'
        )
    else:
        # Default to realSR medium model
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='1conv'
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'params' in checkpoint:
        checkpoint = checkpoint['params']
    elif 'params_ema' in checkpoint:
        checkpoint = checkpoint['params_ema']

    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model

def enhance_image(model, img, device, checkpoint_path):
    """
    Enhances the input image using the loaded SwinIR model.
    Handles images of any size by padding to window_size.
    """
    # Convert to grayscale for document enhancement if needed
    if '004_grayDN' in checkpoint_path or '006_CAR' in checkpoint_path and 'colorCAR' not in checkpoint_path:
        img = img.convert('L')
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = img_np[:, :, None]
    else:
        img_np = np.array(img)

    # Get window size from checkpoint path
    if '006_CAR' in checkpoint_path:
        window_size = 7
    else:
        window_size = 8

    # Determine upscale factor
    if 'x2' in checkpoint_path:
        upscale = 2
    elif 'x3' in checkpoint_path:
        upscale = 3
    elif 'x8' in checkpoint_path:
        upscale = 8
    else:
        upscale = 4

    # Pad image to be divisible by window_size
    h, w = img_np.shape[:2]
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        if len(img_np.shape) == 3:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    if len(img_np.shape) == 3 and img_np.shape[2] == 1:  # Grayscale with channel
        input_tensor = transform(Image.fromarray(img_np[:,:,0])).unsqueeze(0).to(device)
    else:
        input_tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Remove padding and handle upscaling
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    if upscale > 1:  # Super-resolution models
        if pad_h > 0 or pad_w > 0:
            output_tensor = output_tensor[:, :h*upscale, :w*upscale]
    else:  # Denoising or CAR models
        if pad_h > 0 or pad_w > 0:
            output_tensor = output_tensor[:, :h, :w]

    # Convert back to PIL Image
    enhanced_img = transforms.ToPILImage()(output_tensor)
    return enhanced_img

def process_invoice(image_path, checkpoint_path, output_path):
    """
    Processes an invoice image using the specified SwinIR model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_swinir_model(checkpoint_path, device)
    print(f"Model loaded: {checkpoint_path}")
    
    # Process image
    try:
        # For documents, we can either keep color or convert to grayscale
        img = Image.open(image_path)
        if img.mode != 'RGB' and '004_grayDN' not in checkpoint_path:
            img = img.convert('RGB')
        
        # Enhance image
        enhanced_img = enhance_image(model, img, device, checkpoint_path)
        enhanced_img.save(output_path)
        print(f"Enhanced image saved as '{output_path}'")
        return True
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == '__main__':
    # Configuration
    image_path = './input_images/blurredImage2.jpg'
    checkpoint_path = './models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth'
    output_path = 'invoice_enhanced_psnr.jpg'

    process_invoice(image_path, checkpoint_path, output_path)