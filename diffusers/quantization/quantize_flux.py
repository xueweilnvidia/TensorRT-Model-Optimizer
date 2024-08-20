# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import torch
from config import SDXL_FP8_DEFAULT_CONFIG, get_int8_config
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import check_lora, filter_func, load_calib_prompts, quantize_lvl, set_fmha

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

MODEL_ID = {
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
}


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            # negative_prompt=[
            #     "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            # ]
            # * len(prompts),
        ).images


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl-1.0",
        choices=[
            "sdxl-1.0",
            "sdxl-turbo",
            "sd1.5",
            "sd3-medium",
            "flux-dev",
        ],
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Number of denoising steps, for SDXL-turbo, use 1-4 steps",
    )

    # Calibration and quantization parameters
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "fp8"])
    parser.add_argument("--percentile", type=float, default=1.0, required=False)
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
        help=(
            "Ways to collect the amax of each layers, for example, min-max means min(max(step_0),"
            " max(step_1), ...)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=None, help="Will export the ONNX if not None"
    )

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    if args.model == "sd1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16, safety_checker=None
        )
    elif args.model == "sd3-medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16
        )
    elif args.model == "flux-dev":
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    # prompt = "A street musician in his late 30s is frozen in a moment of passionate performance on a busy city corner. His long, dark dreadlocks are caught mid-sway, some falling over his face while others dance in the air around him. His eyes are closed in deep concentration, brows slightly furrowed, as his weathered hands move deftly over the strings of an old, well-loved acoustic guitar. The musician is wearing a vibrant, hand-knitted sweater that's a patchwork of blues, greens, and purples. It hangs loosely over distressed jeans with artistic patches on the knees. On his feet are scuffed brown leather boots, tapping in rhythm with his music. Multiple colorful braided bracelets adorn his wrists, adding to his bohemian appearance. He stands on a gritty sidewalk, with a battered guitar case open at his feet. It's scattered with coins and bills from appreciative passersby, along with a few fallen autumn leaves. Behind him, city life unfolds in a blur of motion: pedestrians hurry past, yellow taxis honk in the congested street, and neon signs begin to flicker to life as dusk settles over the urban landscape. In the foreground, slightly out of focus, a child tugs on her mother's hand, trying to stop and listen to the music. The scene captures the raw energy and emotion of street performance against the backdrop of a bustling, indifferent city."
    prompt = "Create a vibrant street scene at dusk featuring a passionate musician playing a grand piano. The musician, a young woman with flowing hair, is deeply immersed in her music, her hands gracefully dancing over the keys. Surrounding her, the street is lined with warm, glowing street lamps and curious passersby who are captivated by the performance. The background should include charming storefronts and a hint of city skyline, all bathed in the soft hues of sunset"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save("flux-dev-fp16.png")

    # backbone = pipe.unet if args.model != "sd3-medium" else pipe.transformer
    backbone = pipe.transformer

    if args.quant_level == 4.0:
        assert args.format != "int8", "We only support fp8 for Level 4 Quantization"
        assert args.model == "sdxl-1.0", "We only support fp8 for SDXL on Level 4"
        set_fmha(backbone)
    if not args.restore_from:
        # This is a list of prompts
        cali_prompts = load_calib_prompts(
            args.batch_size,
            "./calib/calib_prompts.txt",
        )
        extra_step = (
            1 if args.model == "sd1.5" else 0
        )  # Depending on the scheduler. some schedulers will do n+1 steps
        if args.format == "int8":
            # Making sure to use global_min in the calibrator for SD 1.5
            assert args.collect_method != "default"
            if args.model == "sd1.5":
                args.collect_method = "global_min"
            quant_config = get_int8_config(
                backbone,
                args.quant_level,
                args.alpha,
                args.percentile,
                args.n_steps + extra_step,
                collect_method=args.collect_method,
            )
        elif args.format == "fp8":
            if args.collect_method == "default":
                quant_config = SDXL_FP8_DEFAULT_CONFIG
            else:
                raise NotImplementedError

        def forward_loop(backbone):
            if args.model != "sd3-medium" or "flux-dev":
                pipe.unet = backbone
            else:
                pipe.transformer = backbone
            do_calibrate(
                pipe=pipe,
                calibration_prompts=cali_prompts,
                calib_size=args.calib_size,
                n_steps=args.n_steps,
            )

        # All the LoRA layers should be fused
        check_lora(backbone)
        mtq.quantize(backbone, quant_config, forward_loop)
        mto.save(backbone, f"{args.exp_name}.pt")
    else:
        mto.restore(backbone, args.restore_from)
        # print("skip")
    quantize_lvl(backbone, args.quant_level)
    mtq.disable_quantizer(backbone, filter_func)

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save("flux-dev1-fp8-with-no_proj.png")

    # if you want to export the model on CPU, move the dummy input and the model to cpu and float32
    if args.onnx_dir is not None:
        if args.format == "fp8":
            generate_fp8_scales(backbone)
        modelopt_export_sd(backbone, f"{str(args.onnx_dir)}", args.model)


if __name__ == "__main__":
    main()
