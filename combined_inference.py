import argparse
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from GFPGAN.gfpgan import GFPGANer
import datasets, models, utils
import tempfile

def get_args_parser():
    parser = argparse.ArgumentParser(description='Combined Inference: CrimSniffer and GFPGAN')
    parser.add_argument('--crimsniffer_weight', type=str, required=True, help='Path to load CrimSniffer model weights.')
    parser.add_argument('--gfpgan_weight', type=str, required=True, help='Path to load GFPGAN model weights.')
    parser.add_argument('--input', type=str, required=True, help='Path to read image or folder to be inferenced.')
    parser.add_argument('--output', type=str, required=True, help='Path to save result image.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on.')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the Crimsniffer model.')
    parser.add_argument('--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('--weight', type=float, default=0.5, help='Adjustable weights.')
    return parser.parse_args()

def inference_crimsniffer(model, path_image, device):
    image = datasets.dataloader.load_one_sketch(path_image, simplify=True, device=device).unsqueeze(0).to(device)
    print(f'Loaded image from {path_image}')

    with torch.no_grad():
        result = model(image)
    result = utils.convert.tensor2PIL(result[0])
    return result

def inference_gfpgan(restorer, input_img_path, base_filename, output_folder, args):
    # Ensure the base output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create and ensure the subfolders exist within the base output folder
    # cropped_faces_folder = os.path.join(output_folder, 'cropped_faces')
    Enhanced_faces_folder = os.path.join(output_folder, 'Enhanced_faces')
    Generated_imgs_folder = os.path.join(output_folder, 'Generated_imgs')
    # os.makedirs(cropped_faces_folder, exist_ok=True) # This line should be removed or commented out
    os.makedirs(Enhanced_faces_folder, exist_ok=True)
    os.makedirs(Generated_imgs_folder, exist_ok=True)

    # Process the image
    cropped_faces, Enhanced_faces, Generated_imgs = restorer.enhance(
        input_img_path, # Assuming the enhance method can take a path directly
        has_aligned=args.aligned,
        only_center_face=args.only_center_face,
        paste_back=True,
        weight=args.weight)

    # Debugging: Check number of detected faces
    print(f"Number of cropped faces detected: {len(cropped_faces)}")
    print(f"Number of enhanced faces returned: {len(Enhanced_faces)}")
    print(f"Generated image is {'available' if Generated_imgs is not None else 'not available'}")
    
    # if Enhanced_faces is None:
    #     print('if enhanced images is none')
    # Save restored faces
    for idx, Enhanced_faces in enumerate(Enhanced_faces):
        save_path = os.path.join(Enhanced_faces_folder, f'{base_filename}{idx}.png')
        imwrite(Enhanced_faces, save_path)
        print(f'Saved enhanced image to {save_path}')

    # Save the final restored image
    if Generated_imgs is not None:
        # print('hi there ! in final restored image')
        save_path = os.path.join(Generated_imgs_folder, f'{base_filename}.png')
        imwrite(Generated_imgs, save_path)
        print(f'Saved final restored image to {save_path}')



def main(args): 
    device = torch.device(args.device)
    print(f'Device : {device}')

    # Load CrimSniffer model
    crimsniifer_model = models.CrimSniffer(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=False,
        manifold=args.manifold
    )
    crimsniifer_model.load(args.crimsniffer_weight)
    crimsniifer_model.to(device)
    crimsniifer_model.eval()

    # Load GFPGAN model
    gfpgan_model = GFPGANer(
        model_path=args.gfpgan_weight,
        upscale=args.upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

    # Process input
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    for img_path in img_list:
        # Generate image using CrimSniffer
        generated_image = inference_crimsniffer(crimsniifer_model, img_path, device)

        # Save the generated image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            generated_image.save(temp_file.name)
            temp_file_path = temp_file.name

        # Extract the base filename without extension from the input image path
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Enhance image using GFPGAN
        inference_gfpgan(gfpgan_model, temp_file_path, base_filename, args.output, args)

        # Optionally, delete the temporary file after processing
        os.remove(temp_file_path)

if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    main(args)
