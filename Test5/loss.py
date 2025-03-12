import lpips
import torch
import torchvision.transforms.functional as F


def resize_crop_norm(x, featureExtractor):
    assert isinstance(featureExtractor.size[list(featureExtractor.size.keys())[0]], int)
    out = F.resize(x, featureExtractor.size[list(featureExtractor.size.keys())[0]],
                   interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    out = F.center_crop(out, featureExtractor.size[list(featureExtractor.size.keys())[0]])
    out = F.normalize(out, featureExtractor.image_mean, featureExtractor.image_std)
    return out


def extractFeature(encoder, x, featureName):
    if featureName == 'global':
        out = encoder(x)
        if hasattr(out, 'image_embeds'):
            feat = out.image_embeds
        else:
            feat = out.pooler_output
    elif featureName == 'grid':
        out = encoder(x, output_hidden_states=True, output_attentions=True)
        feat = out.hidden_states[-2]
    elif featureName == 'last_hidden':
        out = encoder(x, output_hidden_states=True, output_attentions=True)
        feat = out.last_hidden_state
    else:
        print(f'{featureName} must be in [global, grid, last_hidden]')
        exit(1)
    return feat


class lpipsLoss:
    def __init__(self, device):
        self.net: lpips.LPIPS = lpips.LPIPS(net='vgg').to(device)
        self.device = device

    def loss(self, x, target, budget):
        l = self.net(x.to(self.device).float(),
                     target.to(self.device).float(),
                     normalize=True)
        return torch.nn.functional.relu(l - budget), l


class AEO:
    def __init__(self, encoder, preprocessor, featureName, device, distance='cos'):
        self.targetAttn = None
        self.targetEmbed = None
        self.imageEncoder = encoder
        self.preprocessor = preprocessor
        self.featureName = featureName
        self.device = device
        self.dtype = torch.float16
        self.imageEncoder = self.imageEncoder.to(device).to(self.dtype)
        self.distance = distance
        if self.distance not in ['mse', 'cos']:
            print(f'{self.distance} must be in [mse, cos]')
            exit(1)

    def loss(self, x, trans=True):  # x should be in [0, 1]
        self.imageEncoder.requires_grad_(False)
        if trans:
            clipX = resize_crop_norm(x, self.preprocessor)
        else:
            clipX = x
        imgEmb = extractFeature(self.imageEncoder, clipX, self.featureName)
        if self.distance == 'mse':
            clipLoss = torch.nn.functional.mse_loss(imgEmb, self.targetEmbed.to(imgEmb.device))
        elif self.distance == 'cos':
            clipLoss = -torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(imgEmb, p=2, dim=-1),
                                                              torch.nn.functional.normalize(
                                                                  self.targetEmbed.to(imgEmb.device), p=2,
                                                                  dim=-1),
                                                              dim=-1).mean()
        else:
            print(f'{self.distance} must be in [mse, cos]')
            exit(1)
        return clipLoss

    def setImgEmbedding(self, x):  # x should be in [0, 1]
        with torch.no_grad():
            clipX = resize_crop_norm(x, self.preprocessor)
            self.targetEmbed = extractFeature(self.imageEncoder, clipX, self.featureName)
