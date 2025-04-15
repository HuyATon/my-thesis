from utils_classes.image_generator import ImageGenerator
import argparse
import torch


# Example:
# python gen.py --ckpt <path> --img <path> --mask <path> --output <path> --batch_size 1   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint inference")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--img", type=str, required=True, help="Path to image folder")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    img_gen = ImageGenerator(
        model_checkpoint_path=args.ckpt,
        img_dir=args.img,
        mask_dir=args.mask,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    with torch.no_grad():
        img_gen.load_model()

        img_gen.create_output_dirs()
        print("Output directories created successfully.")

        img_gen.gen()
        print("Image generation completed.")