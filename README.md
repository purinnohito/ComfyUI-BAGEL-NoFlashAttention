# ComfyUI-Bagel

A custom node package for ComfyUI based on the BAGEL-7B-MoT multimodal model.

## Features

- **Text-to-Image Generation**: Generate high-quality images using natural language prompts.
- **Image Editing**: Edit existing images based on textual descriptions.
- **Image Understanding**: Perform Q&A and analysis on images.
- **Reasoning Process Display**: Optionally display the model's reasoning process.
- **Multiple Image Ratios**: Supports 1:1, 4:3, 3:4, 16:9, 9:16 ratios.
- **Advanced Parameter Control**: Professional parameters like CFG scaling and timestep control.

## Installation Steps

### 1. Model Download
First, download the BAGEL-7B-MoT model:
```bash
# Clone the model using git lfs (recommended)
git lfs install
git clone https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT models/BAGEL-7B-MoT

# Or use huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT', local_dir='models/BAGEL-7B-MoT')"
```

### 2. Dependency Installation
Run the installation script:
```bash
python install.py
```

Or manually install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Restart ComfyUI
Restart ComfyUI to load the new nodes.

## Node Descriptions

### BAGEL Model Loader
The core node for loading BAGEL models.

**Input Parameters**:
- `model_path`: Path to the model (default: "models/BAGEL-7B-MoT")
- `device_map`: Device mapping strategy (auto/single/multi)
- `offload`: Whether to enable memory offloading

**Output**:
- `BAGEL_MODEL`: The loaded model object

### BAGEL Text to Image
Text-to-image generation node.

**Required Parameters**:
- `model`: BAGEL model object
- `prompt`: Text prompt
- `seed`: Random seed (0 for random)
- `image_ratio`: Image aspect ratio
- `cfg_text_scale`: CFG text scaling (1.0-8.0)
- `num_timesteps`: Denoising steps (10-100)

**Optional Parameters**:
- `show_thinking`: Display reasoning process
- `cfg_interval`: CFG interval start value
- `timestep_shift`: Timestep offset
- `cfg_renorm_min`: CFG re-normalization minimum value
- `cfg_renorm_type`: CFG re-normalization type

**Output**:
- `IMAGE`: The generated image
- `STRING`: Reasoning process text (optional)

### BAGEL Image Edit
Image editing node.

**Required Parameters**:
- `model`: BAGEL model object
- `image`: Input image
- `prompt`: Editing prompt
- `seed`: Random seed
- `cfg_text_scale`: CFG text scaling (1.0-8.0)
- `cfg_img_scale`: CFG image scaling (1.0-4.0)
- `num_timesteps`: Denoising steps

**Output**:
- `IMAGE`: The edited image
- `STRING`: Reasoning process text (optional)

### BAGEL Image Understanding
Image understanding node.

**Required Parameters**:
- `model`: BAGEL model object
- `image`: Input image
- `prompt`: Question text

**Optional Parameters**:
- `show_thinking`: Display reasoning process
- `do_sample`: Enable sampling
- `text_temperature`: Text generation temperature (0.0-1.0)
- `max_new_tokens`: Maximum number of new tokens (64-4096)

**Output**:
- `STRING`: Answer text

## Workflow Examples

### Basic Text-to-Image
```
BAGEL Model Loader → BAGEL Text to Image → Save Image
```

### Image Editing Workflow
```
Load Image → BAGEL Image Edit → Save Image
              ↑
BAGEL Model Loader
```

### Image Understanding Workflow
```
Load Image → BAGEL Image Understanding → Preview Text
              ↑
BAGEL Model Loader
```

### Complete Multimodal Workflow
```
BAGEL Model Loader → BAGEL Text to Image → BAGEL Image Edit → BAGEL Image Understanding
                    ↓ (Generate Image)        ↓ (Edit Image)      ↓ (Understand Image)
                 Save Image          Save Image        Preview Text
```

## Usage Tips

### Parameter Tuning
- **CFG Text Scale**: Controls the adherence to the text prompt; higher values mean stricter adherence.
- **CFG Image Scale**: (During editing) Controls the degree of preservation of the original image.
- **Timestep Shift**: Higher values emphasize layout, lower values emphasize details.
- **CFG Renorm Type**: If the generated image is blurry, try using "global".

### Performance Optimization
- Use `device_map="auto"` to automatically allocate GPU memory.
- Enable `offload=True` to unload the model to disk when memory is insufficient.
- Fewer `num_timesteps` can speed up generation but may reduce quality.

### Prompt Suggestions
- **Image Generation**: Describe the scene, style, color, lighting, etc., in detail.
- **Image Editing**: Clearly specify the parts to be modified and the desired effects.
- **Image Understanding**: Ask specific questions about the content, emotion, technical details, etc.

## System Requirements

- **GPU**: Recommended 24GB+ VRAM (can use memory offloading to reduce requirements)
- **RAM**: 16GB+ system memory
- **Storage**: About 15GB of free space for model files
- **Python**: Version 3.8+
- **CUDA**: Version 11.8+ (recommended)

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check if the model path is correct
   - Ensure the model files are fully downloaded
   - Check if there is enough storage space

2. **Insufficient Memory**
   - Enable `offload=True`
   - Use `device_map="single"` for single GPU mode
   - Reduce memory usage of other applications

3. **Poor Image Quality**
   - Adjust CFG parameters
   - Increase the number of timesteps
   - Optimize the prompt description

4. **Import Errors**
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check CUDA version compatibility
   - Restart ComfyUI

### Log Viewing
Check the ComfyUI console output for detailed error information.

## Changelog

### v1.0.0
- Initial release
- Supports text-to-image generation, image editing, and image understanding
- Complete parameter control and error handling
- Memory optimization and multi-GPU support

## Related Links

- [BAGEL Official Paper](https://arxiv.org/abs/2505.14683)
- [BAGEL Model Homepage](https://bagel-ai.org/)
- [Hugging Face Model](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- [Online Demo](https://demo.bagel-ai.org/)
- [Discord Community](https://discord.gg/Z836xxzy)

## License

This project is licensed under the Apache 2.0 License. Please refer to the official license terms for the use of the BAGEL model.

## Contribution

Contributions are welcome! Please submit issue reports and feature requests. If you wish to contribute code, please create an issue to discuss your ideas first.
