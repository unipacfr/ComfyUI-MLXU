# Fix for ASDX_MLXSampler VAE Parameter Issue

## Problem
The ASDX_MLXSampler node was failing with:
```
TypeError: ASDX_MLXSampler.execute() got an unexpected keyword argument 'vae'
```

This occurred because ComfyUI was passing a VAE parameter to the execute method, but the method signature didn't include this parameter.

## Solution
Added the missing `vae` parameter to the ASDX_MLXSampler.execute() method signature in `/Volumes/X10Pro/Images/Projet/ComfyUI-MLXU/apple_silicon_nodes/sampler/__init__.py`:

```python
# Added this parameter at line 149:
vae: Any | None = None,
```

## Result
- ✅ The TypeError is now resolved
- ✅ The node can accept VAE parameters without crashing
- ✅ All existing functionality remains intact

## Verification
The fix was minimal and surgical:
1. Added only the missing parameter to satisfy ComfyUI's parameter passing system
2. Made it optional (default None) for backward compatibility
3. Follows the same pattern used by other nodes in the codebase (ASDX_VAEDecode, ASDX_VAEEncode)
4. Does not change any existing functionality

## Status
The original issue has been resolved. The ASDX_MLXSampler now properly accepts VAE parameters and will not crash with the TypeError anymore.

## Update (2026-09-07)
The `vae` parameter is no longer a no-op. It is now consumed by the Krea2
Identity Edit **pixel path** (`_SamplerCore._prepare_krea2_identity_edit_pixels`):
when `source_image` + `vae` are both wired, the sampler fits the source image in
pixel space and VAE-encodes it itself (the blur-proof path the reference and the
v1_2 LoRA were trained with), instead of fitting a pre-encoded latent. See the
canon record `Krea2 Identity Edit source is fitted in pixel space, never in
latent space` and commit `08fcd27`. The "does not change any existing
functionality" note above described the original minimal fix only; the parameter
now has a real consumer.