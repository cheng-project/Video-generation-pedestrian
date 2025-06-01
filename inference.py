import os
import torch
import argparse
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif

def load_lora_for_unet_spatiotemporal(
        base_model_path="",
        lora_path=None,
        device="cuda",
        torch_dtype=torch.float16,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        use_safetensors=True
):
    """
    Load LoRA weights for UNetSpatioTemporalConditionModel

    parameter:
        base_model_path: Base UNet model path (HF hub or local path)
        lora_path: LoRA weight path
        device:  ("cuda" or "cpu")
        torch_dtype: data type
        r: LoRA rank
        lora_alpha: LoRA alpha value
        lora_dropout: LoRA dropout rate
        target_modules: The target module for applying LoRA
        use_safetensors:
    """
    # 1. Load the basic UNet model
    print(f"Loading base UNet model from {base_model_path}...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        variant="fp16",
        low_cpu_mem_usage=True
    )

    # 2. Create LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        inference_mode=False
    )

    # 3. Inject the LoRA adapter into the model
    print("Injecting LoRA adapters into the model...")
    lora_unet = inject_adapter_in_model(lora_config, unet)

    # 4. If there is a LoRA weight path, load the weights.
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")

        # Check if it is a safetensors file.
        if use_safetensors and lora_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location="cpu")

        # Load the weights into the model.
        set_peft_model_state_dict(lora_unet, lora_state_dict, adapter_name="default")

    lora_unet.to(device)
    print(f"Model moved to {device}")

    return lora_unet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable Video Diffusion."
    )
    parser.add_argument(
        "--lora_weights_path",
        default="weights/pytorch_lora_weights.safetensors",
        type=str,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--test_jpg_path",
        type=str,
        default="frame_000000.jpg",
        help="test_jpg_path",
    )

    parser.add_argument(
        "--out_file_path",
        type=str,
        default="frame_000000.git",
        help="out_file_path",
    )

    parser.add_argument(
        "--seed", type=int, default=123, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help=("The global scaling factor that controls the adaptability strength"),
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
        help=("Decompress video frames in blocks to reduce memory usage"),
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
        help=("Control the intensity of the movement in the video "
              "(the higher the value, the more intense the movement)."),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help=("The frame rate of the output video (frames per second) only affects the playback speed "
              "and does not change the content."),
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help=("Controlling the intensity of noise enhancement in the input image affects the degree of difference "
              "between the video and the input image."),
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # example
    lora_unet = load_lora_for_unet_spatiotemporal(
        lora_path="weights/pytorch_lora_weights.safetensors",
        r=args.rank,
        lora_alpha=args.lora_alpha,
    )

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        unet=lora_unet,
        low_cpu_mem_usage=False,
        variant="fp16",
        local_files_only=True,
    )
    pipeline.to("cuda:0")

    height = args.height
    width = args.width
    test_jpg_path = args.output_dir
    num_frames = args.num_frames
    generator = torch.manual_seed(args.seed)
    with torch.inference_mode():
        video_frames = pipeline(
            load_image(test_jpg_path).resize((width, height)),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            motion_bucket_id=args.motion_bucket_id,
            fps=args.fps,
            noise_aug_strength=args.noise_aug_strength,
            # generator=generator,
        ).frames[0]


    for i in range(num_frames):
        img = video_frames[i]
        # video_frames[i] = np.array(img)
        video_frames[i] = img
    export_to_gif(video_frames, args.out_file_path, args.fps)



if __name__ == "__main__":
    main()









