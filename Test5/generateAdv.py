import argparse
import os
import shutil

import numpy as np
import prodigyopt
import torch
import torchvision.transforms.functional as F
from PIL import Image
from advertorch.attacks.utils import rand_init_delta
from torch.cuda.amp import autocast, GradScaler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel, SiglipImageProcessor

import Masked_PGD
import loss


class AdamWAttack:
    def __init__(self, lossF, budget, steps, lr, clipMin, clipMax, randInit, bound):
        self.lossF = lossF
        self.budget = budget
        self.steps = steps
        self.lr = lr
        self.clipMin = clipMin
        self.clipMax = clipMax
        self.randInit = randInit
        self.bound = bound
        self.lpLoss = None
        if self.bound == 'lpips':
            self.lpLoss = loss.lpipsLoss(self.lossF.device)

    def perturb(self, x, nouse, verbose=False):
        assert x.shape[0] == 1
        advs = x.clone().detach().to(torch.float32)
        ori = x.clone().detach().to(torch.float32)
        bestAdvs = advs.clone().detach().to(torch.float32)
        bestL = float('inf')
        if self.randInit:
            delta = torch.zeros_like(advs)
            rand_init_delta(
                delta, advs, np.inf, self.budget, self.clipMin, self.clipMax)
            advs = advs + delta
        opt = torch.optim.AdamW([advs], lr=self.lr, weight_decay=0)
        scaler = GradScaler()
        for i in range(self.steps):
            gradContainer = advs.clone().detach()
            gradContainer.requires_grad = True
            gradContainer.grad = None
            advs.grad = None
            opt.zero_grad()
            with autocast(dtype=lossF.dtype):
                l = self.lossF.loss(gradContainer)
            gradContainer.grad = None
            scaler.scale(l).backward()
            if self.bound == 'lpips':
                advs.requires_grad = True
                with autocast(dtype=torch.float32):
                    lpl, lpNum = self.lpLoss.loss(advs, ori, self.budget)
                scaler.scale(lpl).backward()
                lpGradNorm = torch.linalg.vector_norm(advs.grad, dim=(-3, -2, -1))
                if lpGradNorm > 0:
                    advs.grad = advs.grad / lpGradNorm * torch.linalg.vector_norm(gradContainer.grad, dim=(-3, -2, -1))
                advs.grad += gradContainer.grad
            else:
                advs.grad = gradContainer.grad
            scaler.step(opt)
            scaler.update()
            advs.requires_grad = False
            if self.bound != 'lpips':
                advs[:, :, :, :] = torch.clamp(advs.clone().detach(), ori - self.budget, ori + self.budget)
                with torch.no_grad():
                    lpNum = torch.mean(torch.abs(advs.clone() - ori.clone()))
            advs[:, :, :, :] = torch.clamp(advs.clone().detach(), self.clipMin, self.clipMax)
            if verbose:
                with torch.no_grad():
                    print(
                        f'\rstep{i}: loss: {l.item()} zero_grad: {(advs.grad.data == 0.0).to(torch.int).sum()} lpips: {lpNum.item()}',
                        end='')
            if l.item() < bestL:
                bestAdvs = advs.clone().detach()
                bestL = l.item()

        return bestAdvs


class prodigyAttack:
    def __init__(self, lossF, budget, steps, lr, clipMin, clipMax, randInit, bound):
        self.lossF = lossF
        self.budget = budget
        self.steps = steps
        self.lr = lr
        self.clipMin = clipMin
        self.clipMax = clipMax
        self.randInit = randInit
        self.bound = bound
        self.lpLoss = None
        if self.bound == 'lpips':
            self.lpLoss = loss.lpipsLoss(self.lossF.device)

    def perturb(self, x, nouse, verbose=False):
        assert x.shape[0] == 1
        advs = x.clone().detach().to(torch.float32)
        ori = x.clone().detach().to(torch.float32)
        bestAdvs = advs.clone().detach().to(torch.float32)
        bestL = float('inf')
        if self.randInit:
            delta = torch.zeros_like(advs)
            rand_init_delta(
                delta, advs, np.inf, self.budget, self.clipMin, self.clipMax)
            advs = advs + delta
        opt = prodigyopt.Prodigy([advs], lr=self.lr, weight_decay=0)
        scaler = GradScaler()
        for i in range(self.steps):
            gradContainer = advs.clone().detach()
            gradContainer.requires_grad = True
            gradContainer.grad = None
            advs.grad = None
            opt.zero_grad()
            with autocast(dtype=lossF.dtype):
                l = self.lossF.loss(gradContainer)
            gradContainer.grad = None
            scaler.scale(l).backward()
            if self.bound == 'lpips':
                advs.requires_grad = True
                with autocast(dtype=torch.float32):
                    lpl, lpNum = self.lpLoss.loss(advs, ori, self.budget)
                scaler.scale(lpl).backward()
                lpGradNorm = torch.linalg.vector_norm(advs.grad, dim=(-3, -2, -1))
                if lpGradNorm > 0:
                    advs.grad = advs.grad / lpGradNorm * torch.linalg.vector_norm(gradContainer.grad, dim=(-3, -2, -1))
                advs.grad += gradContainer.grad
            else:
                advs.grad = gradContainer.grad
            scaler.step(opt)
            scaler.update()
            advs.requires_grad = False
            if self.bound != 'lpips':
                advs[:, :, :, :] = torch.clamp(advs.clone().detach(), ori - self.budget, ori + self.budget)
                with torch.no_grad():
                    lpNum = torch.mean(torch.abs(advs.clone() - ori.clone()))
            advs[:, :, :, :] = torch.clamp(advs.clone().detach(), self.clipMin, self.clipMax)
            if verbose:
                with torch.no_grad():
                    print(
                        f'\rstep{i}: loss: {l.item()} zero_grad: {(advs.grad.data == 0.0).to(torch.int).sum()} lpips: {lpNum.item()}',
                        end='')
            if l.item() < bestL:
                bestAdvs = advs.clone().detach()
                bestL = l.item()

        return bestAdvs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir', type=str, help='/path/to/the/folder/containing/clean/images')
    parser.add_argument('--saveDir', type=str, help='/path/to/save')
    parser.add_argument('--model', type=str, help='/path/to/the/image/encoder/')
    parser.add_argument('--targetP', type=str, help='The path of target image\'s dir')
    parser.add_argument('--opt', type=str, help='pgd, adamw, or prodigy')
    parser.add_argument('--budget', type=float, help='budget if bound == lpips else budget / 255')
    parser.add_argument('--steps', type=int, help='The larger, the better')
    parser.add_argument('--stepsize', type=float, help='Typically, 1.0 for pgd and prodigy, 1e-3 for adamw')
    parser.add_argument('--gpu', type=int, help='Just the GPU index')
    parser.add_argument('--dist', type=str, help='mse or cos. cos is usually better')
    parser.add_argument('--fn', type=str, help='global, grid, or last_hidden')
    parser.add_argument('--bound', type=str, help='lpips or linf')
    args = parser.parse_args()
    print(args)
    imgDir = args.imgDir
    saveDir = args.saveDir
    modelN = args.model
    targetP = args.targetP
    budget, steps, stepsize = args.budget, args.steps, args.stepsize
    gpu = args.gpu
    optN = args.opt
    dist = args.dist
    fn = args.fn
    bound = args.bound
    # set cuda
    device = torch.device("cuda:{}".format(gpu))
    torch.cuda.set_device(device)
    # load model
    try:
        encoder = CLIPVisionModelWithProjection.from_pretrained(modelN)
        preprocessor = CLIPImageProcessor.from_pretrained(modelN)
    except:
        print('Not CLIP but SigLip?')
        encoder = SiglipVisionModel.from_pretrained(modelN)
        preprocessor = SiglipImageProcessor.from_pretrained(modelN)
    lossF = loss.AEO(
        encoder,
        preprocessor,
        fn,
        device,
        dist
    )
    if optN == 'pgd':
        attack = Masked_PGD.LinfPGDAttack(lossF.loss,
                                          # [-1, 1] and loss value descent. If you claim an optimizer better than PGD, try.
                                          lambda x, y: x,
                                          eps=budget / 255,  # budget when pixel values are within [0, 1]
                                          nb_iter=steps,
                                          rand_init=True,
                                          eps_iter=stepsize / 255,
                                          clip_min=0, clip_max=1, targeted=True)
    elif optN == 'adamw':
        attack = AdamWAttack(lossF,
                             budget if bound == 'lpips' else budget / 255,  # budget when pixel values are within [0, 1]
                             steps,
                             stepsize,
                             0, 1,
                             False if bound == 'lpips' else True, bound
                             )
    elif optN == 'prodigy':
        attack = prodigyAttack(lossF,
                               budget if bound == 'lpips' else budget / 255,
                               # budget when pixel values are within [0, 1]
                               steps,
                               stepsize,
                               0, 1,
                               False if bound == 'lpips' else True, bound
                               )
    else:
        print(f'{optN} must be in [pgd, adamw, prodigy]')
        exit(1)
    # set directory
    saveDir = os.path.join(saveDir,
                           imgDir.split('/')[-1] if imgDir.split('/')[-1] != '' else imgDir.split('/')[-2],
                           targetP.split('/')[-1] if targetP.split('/')[-1] != '' else targetP.split('/')[-2],
                           '{}_{}'.format(
                               modelN.split('/')[-1] if modelN.split('/')[-1] != '' else modelN.split('/')[-2],
                               fn
                           ),
                           '{}_{}_dist{}_budget{}_steps{}_stepSize{}'.format(bound,
                                                                             optN, dist,
                                                                             budget,
                                                                             steps,
                                                                             stepsize
                                                                             )
                           )
    if os.path.exists(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir, exist_ok=True)
    # attacking
    for targetF in os.listdir(targetP):
        targetImg = Image.open(os.path.join(targetP, targetF)).convert('RGB')
        lossF.setImgEmbedding(F.to_tensor(targetImg).unsqueeze(0).to(device).to(lossF.dtype))
        for imgF in os.listdir(imgDir):
            if '.png' not in imgF and '.jpg' not in imgF:
                continue
            img = F.to_tensor(Image.open(os.path.join(imgDir, imgF)).convert('RGB')).unsqueeze(0).to(device).to(
                torch.float32)
            fakeY = torch.zeros((img.shape[0],), device=device, dtype=torch.int)
            with torch.no_grad():
                testImg = lossF.preprocessor.preprocess(
                    images=Image.open(os.path.join(imgDir, imgF)).convert('RGB'), do_resize=True,
                    return_tensors="pt", do_convert_rgb=True
                ).pixel_values
                l = lossF.loss(testImg.to(device).to(lossF.dtype), trans=False)
                print(f'\nInitial Loss: {l.item()}')
            advs = attack.perturb(img, fakeY, verbose=True)
            for adv in advs:
                F.to_pil_image(torch.clamp(adv, 0, 1)).save(
                    os.path.join(saveDir, '{}_{}.png'.format(os.path.splitext(imgF)[0], os.path.splitext(targetF)[0])))
            with torch.no_grad():
                testImg = lossF.preprocessor.preprocess(
                    images=Image.open(os.path.join(saveDir, '{}_{}.png'.format(os.path.splitext(imgF)[0],
                                                                               os.path.splitext(targetF)[0]))).convert(
                        'RGB'),
                    do_resize=True,
                    return_tensors="pt", do_convert_rgb=True
                ).pixel_values
                l = lossF.loss(testImg.to(device).to(lossF.dtype), trans=False)
                print(f'\nFinal Loss: {l.item()}')
