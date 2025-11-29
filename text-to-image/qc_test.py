from stable_diff_trt import *

prompt = "Neon cyberpunk street market at night"

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

img_baseline, tr_baseline = test_n_stable_diffusion(config_baseline, 2)

subset = tr_baseline[1:]

avg = {
    key: sum(d[key] for d in subset) / len(subset)
    for key in subset[0]
}

print(avg)

breakpoint()