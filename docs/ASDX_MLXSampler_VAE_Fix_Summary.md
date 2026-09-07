# ASDX_MLXSampler VAE Parameter Fix - Summary

## Original Problem
The ASDX_MLXSampler node was failing with:
```
TypeError: ASDX_MLXSampler.execute() got an unexpected keyword argument 'vae'
```

## Solution Implemented
Added the missing `vae` parameter to the ASDX_MLXSampler.execute() method signature in `/Volumes/X10Pro/Images/Projet/ComfyUI-MLXU/apple_silicon_nodes/sampler/__init__.py`:

```python
# Added this parameter at line 149:
vae: Any | None = None,
```

## Results
✅ **Fixed the TypeError**: The node no longer crashes when VAE parameters are passed  
✅ **Maintains backward compatibility**: Parameter is optional with default None  
✅ **Follows established patterns**: Matches the same approach used by ASDX_VAEDecode and ASDX_VAEEncode nodes  
✅ **Minimal surgical change**: Only the necessary fix was implemented  

## Verification
- The fix resolves the original error completely
- All existing functionality remains intact
- No breaking changes to the API

## Related Issue (Unrelated to Original Request)
A new error has emerged in Krea2 Identity Edit functionality:
```
RuntimeError: ASDX: Identity Edit prep failed: [broadcast_shapes] Shapes (1,799,94,168) and (1,16,1,1) cannot be broadcast.
```

This is a separate bug in the Krea2 Identity Edit feature that was not part of the original request and requires a different investigation.

**Update (2026-09-07):** that Krea2 Identity Edit bug is now resolved. The
`vae` parameter added above is consumed by the new pixel path
(`_prepare_krea2_identity_edit_pixels`), which fits the source image in pixel
space and VAE-encodes it -- the geometry the v1_2 LoRA was trained with. The
broadcast error (a `[B,H,W,C]` latent misread as `[B,C,H,W]`) and the downstream
face artifacts both came from fitting in latent space; the pixel path removes
that. See the canon record `Krea2 Identity Edit source is fitted in pixel space,
never in latent space` and commit `08fcd27`.

## Status
The original VAE parameter issue has been successfully resolved. The ASDX_MLXSampler now properly accepts VAE parameters and will not crash with the TypeError anymore. The parameter is now actively used by the Krea2 Identity Edit pixel path (see Update above).