

# Video-generation-pedestrian

**Lora fine-tuning of the pedestrian dataset based on Stable Video Diffusion 🚀**


## Part: Inference

### Result
| Init Image                                                                                           | Fine-tuning                                                                                         ||
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-|
| ![demo](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example01_resize.jpg) | ![ori](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example01.gif) | |
| ![demo](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example02_resize.jpg) | ![ori](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example02.gif) | |
| ![demo](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example03_resize.jpg) | ![ori](https://github.com/cheng-project/Video-generation-pedestrian/blob/main/.asset/example03.gif) | |


### Inference Configuration
```bash
python3 inference.py \
    --lora_weights_path='weights/pytorch_lora_weights.safetensors' \
    --num_frames=25 \
    --test_jpg_path='frame.jpg' \
    --out_file_path="frame.gif" \
    --width=512 \
    --height=320 \
    --rank=4 \
    --lora_alpha=4 \
    --decode_chunk_size=8 \
    --motion_bucket_id=25 \
    --fps=7 \
    --noise_aug_strength=0.02
```




  

