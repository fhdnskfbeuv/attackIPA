import argparse
import os.path
import shutil

import torch.cuda
from PIL import Image
from diffusers import AutoPipelineForText2Image, UniPCMultistepScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    print(args)
    gpu = args.gpu
    # set cuda
    device = torch.device("cuda:{}".format(gpu))
    torch.cuda.set_device(device)
    # load model
    pipe: AutoPipelineForText2Image = AutoPipelineForText2Image.from_pretrained('acheong08/SD-V1-5-cloned').to(device) # runway delete their sd-v1-5 :(
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models', weight_name='ip-adapter_sd15.safetensors')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(0.75)
    pipe.safety_checker = None
    # load prompt
    with open('./stable_diffusion_prompts_small', 'r') as f:
        prompts = f.readlines()
    saveDir = './inferResult'
    if os.path.exists(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir, exist_ok=True)
    for imgF in open('./answer.txt', 'r').read().splitlines():
        if '.png' not in imgF and '.jpg' not in imgF:
            continue
        img = Image.open(os.path.join('./', imgF)).convert('RGB')
        for i, p in enumerate(prompts):
            out = pipe.__call__(prompt='(best quality, ultra highres:1.2), ' + p,
                                num_inference_steps=30,
                                guidance_scale=7.5,
                                negative_prompt="(worst quality:2.0)",
                                ip_adapter_image=img
                                ).images[0]
            out.save(os.path.join(saveDir, '{}_{}.png'.format(os.path.splitext(imgF)[0], i)))