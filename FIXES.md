# LTX-Video Inference Fixes

This document details the fix applied to resolve the critical issue with the LTX-Video inference pipeline when using the development configuration with prompt enhancement.

## Issue Fixed

### Florence-2 Model Initialization Error (DaViT Issue)

**Error**: `AttributeError: 'DaViT' object has no attribute '_initialize_weights'`

**Root Cause**: The Florence-2 prompt enhancement model (`MiaoshouAI/Florence-2-large-PromptGen-v2.0`) uses a DaViT (Data-efficient image Transformer) architecture component. In transformers version 4.52.0 and later, there was a breaking change that caused the DaViT model to fail initialization due to a missing `_initialize_weights` method.

**Reference**: [ComfyUI-LTXVideo Issue #232](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/232)

## Solution

**Fix**: Downgrade transformers to version <4.52.0

**Command**:
```bash
pip install 'transformers<4.52.0'
```

**Explanation**: 
- The issue was introduced in transformers v4.52.0 
- Versions prior to 4.52.0 handle the DaViT initialization correctly
- This is a clean fix that allows the prompt enhancement to work as intended without code modifications

## Verification

After applying the fix, the Florence-2 model loads successfully:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# This now works without errors
model = AutoModelForCausalLM.from_pretrained(
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0', 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0', 
    trust_remote_code=True
)
```

## Impact of the Fix

### Before Fix
- **Error with dev config**: Inference would crash with `AttributeError` when trying to load Florence-2 model
- **Limited to distilled model**: Users could only use the lower-quality distilled configuration
- **No prompt enhancement**: Even when working, prompt enhancement features were unavailable

### After Fix
- **Full dev config support**: Can now use the full development model configuration with better guidance scales
- **Prompt enhancement working**: The Florence-2 and Llama models load and enhance prompts automatically
- **Better quality output**: Access to multi-scale pipeline with proper inference steps (27+13 vs 7+3)
- **Automatic prompt enhancement**: Short prompts get enhanced with cinematic details

## Configuration Differences

### Distilled Config (`ltxv-13b-0.9.7-distilled.yaml`)
- Simple inference: 7 steps first pass, 3 steps second pass  
- No guidance scaling: `guidance_scale: 1` (flat)
- Prompt enhancement disabled: `prompt_enhancement_words_threshold: 0`

### Dev Config (`ltxv-13b-0.9.7-dev.yaml`) - Now Working
- Complex inference: 30 steps with skip optimizations (effective 27+13)
- Dynamic guidance scaling: `[1, 1, 6, 8, 6, 1, 1]` and `[0, 0, 4, 4, 4, 2, 1]`
- Prompt enhancement enabled: `prompt_enhancement_words_threshold: 120`
- CFG-star rescaling enabled for better quality

## Testing Results

**Command Used**:
```bash
python inference.py \
  --prompt "A woman with brown hair wearing a pink sweater" \
  --height 512 --width 768 --num_frames 25 --frame_rate 24 --seed 2025 \
  --pipeline_config configs/ltxv-13b-0.9.7-dev.yaml
```

**Results**:
- ✅ Florence-2 and Llama models load successfully  
- ✅ Prompt gets automatically enhanced with cinematic details
- ✅ Generates video using dev config's superior inference pipeline
- ✅ Output: `video_output_0_a-woman-with-brown-hair-wearing-a_2025_512x768x25_0.mp4`
- ✅ Full prompt enhancement pipeline working as intended

The fix enables users to benefit from the complete LTX-Video development pipeline including automatic prompt enhancement for better quality results.