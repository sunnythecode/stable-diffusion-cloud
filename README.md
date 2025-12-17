1. Copy libraries from environment.yaml
2. Run `python text-to-image/qc_test.py`


Sample config for inference:
```
config_baseline = PromptConfig(
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted, ugly",
    use_negative_prompt=False,
    num_inference_steps=20,
    height=768,
    width=768,
    guidance_scale=7.5,
    seed=1,
    scheduler_name="default",
    use_fp16=True,
    use_tensorrt=True,
    output_filename=f"{prompt}_noneg.png"
)
```
Some things: keep tensorrt on, use fp16, use 512x512, at the moment turn negative prompt off
Guidance scale is how closely model follows prompt