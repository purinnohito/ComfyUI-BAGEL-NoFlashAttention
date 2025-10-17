# Windows Optimization Guide for ComfyUI-BAGEL

## Performance vs Precision Trade-off

This modified version of ComfyUI-BAGEL is optimized for Windows without flash_attn.

### Configuration Options

Edit `flash_attn_compat.py` to adjust the trade-off:

```python
# At the top of flash_attn_compat.py
USE_HIGH_PRECISION = False  # Change to True for maximum quality (slower)
```

### Settings:

**USE_HIGH_PRECISION = False** (Default, Recommended)
- ✅ Fast execution using PyTorch SDPA
- ✅ Low memory usage
- ✅ Good quality for most tasks
- ⚠️ May have minor precision differences vs flash_attn

**USE_HIGH_PRECISION = True** (Maximum Quality)
- ✅ Maximum numerical precision
- ✅ Closest to original flash_attn behavior
- ❌ Much slower
- ❌ Higher memory usage
- ❌ May cause "Allocation on device" errors

### Memory Optimization Tips

If you encounter "Allocation on device" errors:

1. **Reduce batch size** in your workflow
2. **Use smaller images** (resize before processing)
3. **Set USE_HIGH_PRECISION = False**
4. **Close other applications** to free GPU memory
5. **Use torch.cuda.empty_cache()** between operations

### Attention Implementation

The code automatically uses:
- **flash_attn**: If available (Linux with CUDA)
- **SDPA (fast)**: If USE_HIGH_PRECISION=False (Windows default)
- **Manual (precise)**: If USE_HIGH_PRECISION=True

### Troubleshooting

**Problem**: Images are corrupted or low quality
**Solution**: Set `USE_HIGH_PRECISION = True` in `flash_attn_compat.py`

**Problem**: Out of memory errors
**Solution**: Set `USE_HIGH_PRECISION = False` and reduce image size

**Problem**: Very slow processing
**Solution**: Ensure `USE_HIGH_PRECISION = False` and `_attn_implementation="sdpa"`

### Performance Comparison

|  | Speed | Memory | Quality |
|---|---|---|---|
| flash_attn (original) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| SDPA (USE_HIGH_PRECISION=False) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Manual (USE_HIGH_PRECISION=True) | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |