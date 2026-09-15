import torch
import os
import json
import time
import random
import torch.nn as nn
from torch.nn import Module
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
from typing import Dict, Tuple, Any, Union, Optional, Callable, List
from layer_selector import select_best_layer_combination
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nibabel as nib
from get_in_class import get_in_class_text
from torchvision import transforms as trn
from monai.transforms import CropForeground

class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample.get_fdata())
    
class AddChannelDim:
    def __call__(self, sample, dim=0):
        return sample.unsqueeze(dim=dim)

class ToFloat:
    def __call__(self, sample):
        return sample.float()

def empty_threshold(x):
    return x > 0

to_np = lambda x: x.data.cpu().numpy()

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, preprocess, dataset_dir):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        path = self.image_paths[idx]
        if "oasis" in self.dataset_dir: 
            npath = '/'.join(path.split('/')[:-1]) + '/preprocessed-' + path.split('/')[-1][:-6] + 'pt'
            img = torch.load(npath, weights_only = False, map_location = 'cpu').squeeze(0)
            return img, path

        else:
            img = Image.open(path).convert("RGB")
            img = self.preprocess(img)

        return img, path

def register_hooks(model: Module) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Automatically registers hooks for intermediate feature extraction 
    based on whether the model is from Hugging Face's `transformers` 
    or OpenAI's `clip` library.
    """
    intermediate_features: Dict[str, torch.Tensor] = {}

    def hook_fn(layer_name: str) -> Callable[[Module, Tuple[Any, ...], torch.Tensor], None]:
        """
        Creates a hook function to store intermediate activations.
        Args:
            layer_name (str): The name of the layer being hooked.

        Returns:
            Callable: A hook function.
        """
        def hook(module: Module, input: Tuple[Any, ...], output: torch.Tensor) -> None:
            intermediate_features[layer_name] = output
        return hook

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    # **CASE 1: UnimedClip (ResNet-based models)**

    if hasattr(model.visual, 'trunk'):
        print("Detected: BiomedCLIP")
        for i, layer in enumerate(model.visual.trunk.blocks):
            hook = layer.register_forward_hook(hook_fn(f'layer_{i+1}'))
            hooks.append(hook)
    else:
        print("Detected: Unimedclip")
        for i, layer in enumerate(model.visual.transformer.resblocks):
            hook = layer.register_forward_hook(hook_fn(f'layer_{i+1}'))
            hooks.append(hook)
    
    return intermediate_features, hooks

def process_intermediate_features(model: Module, intermediate_features: Dict[str, torch.Tensor]) -> torch.Tensor:

    stacked_features: List[torch.Tensor] = []

    if hasattr(model.visual, 'transformer'):
        for i in range(len(model.visual.transformer.resblocks)):
            layer_key = f'layer_{i+1}'
            if layer_key in intermediate_features:
                feature = intermediate_features[layer_key][0] 
                feature = model.visual.ln_post(feature) 
                feature = feature @ model.visual.proj            
                stacked_features.append(feature)


    if hasattr(model.visual, 'trunk'):
        for i in range(len(model.visual.trunk.blocks)):
            layer_key = f'layer_{i+1}'
            if layer_key in intermediate_features:
                feature = intermediate_features[layer_key][0] 
                feature = model.visual.trunk.norm(feature) 
                feature = model.visual.head(feature[0,:].unsqueeze(0))            
                stacked_features.append(feature)

    return torch.stack(stacked_features, dim=1)  # Stack intermediate features



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def get_scores_prob(args, probs):

    if args.score == 'MCM':
        probs = -np.max(probs, axis=-1)
    if args.score == 'energy':
        probs_torch = torch.from_numpy(probs).float()
        probs = -torch.logsumexp(probs_torch, dim=-1).numpy()
    if args.score == 'entropy':
        probs =  entropy(probs, axis=-1) 
    if args.score == 'var':
        probs = -np.var(probs, axis=-1)
    if args.score == 'MOD':
        probs = -np.max(probs, axis=-1)
    return probs

def get_mod_scores(args, in_images, ood_images, model, preprocess, tokenizer):

    in_class = get_in_class_text(args)

    if "biomed" in args.models_name:
        context_length = 256
        text = tokenizer([x for x in in_class], context_length=context_length).cuda()
    else:
        text = tokenizer([x for x in in_class]).cuda()

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


    batch_size = 64   # or 8
    num_workers = 16   # 
    
    if 'oasis' in args.dataset_dir:
        crop_dim = 224
        preprocess = trn.Compose([tio.ToCanonical(), ToTensor(), AddChannelDim(), CropForeground(select_fn=empty_threshold, margin=0,  k_divisible=[crop_dim, crop_dim, crop_dim]), tio.CropOrPad((crop_dim,crop_dim,crop_dim)), ToFloat(), tio.ZNormalization()])
    
    in_dataset = ImagePathDataset(in_images, preprocess, args.dataset_dir)
    ood_dataset = ImagePathDataset(ood_images, preprocess, args.dataset_dir)

    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    intermediate_features, hooks = register_hooks(model)
    all_logits_for_selection = []

    with torch.no_grad():
        for image_batch, path in tqdm(in_loader):
            image_batch = image_batch.cuda()

            _ = model.encode_image(image_batch)
            stacked_features = process_intermediate_features(model, intermediate_features)
            image_features =  stacked_features.float()            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if "quickgelu" in args.models_name:
                logits = image_features @ text_features.t()

            if "biomed" in args.models_name:
                logits = model.logit_scale * image_features @ text_features.t()

            all_logits_for_selection.append(logits.detach().cpu())
            
    all_logits_tensor = torch.vstack(all_logits_for_selection)
    t0 = time.perf_counter()
    best_layers, layer_optimization_summary = select_best_layer_combination(args, id_logits=all_logits_tensor, temperature=1.0, max_layer_idx=None, reference_layer=-1, max_length=args.number_of_mod_layers, sample_size=len(in_images), n_jobs=-1)
    print(f"{time.perf_counter()-t0:.4f}s")
    
    ood_score = []
    ind_score = []
    
    if args.manual_input_best_layers:
        best_layers = [args.input_best_layers]
    
    print(best_layers)


    with torch.no_grad():
        for image_batch, path in tqdm(ood_loader):
            image_batch = image_batch.cuda()
                
            _ = model.encode_image(image_batch)
            stacked_features = process_intermediate_features(model, intermediate_features)
            image_features =  stacked_features.float()            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if "quickgelu" in args.models_name:
                logits = image_features @ text_features.t()

            if "biomed" in args.models_name:
                logits = model.logit_scale * image_features @ text_features.t()

            probs_batch = to_np(logits)            
            probs_batch = probs_batch[:, best_layers, :]
            for b in range(probs_batch.shape[0]):
                ood_score.append(get_scores_prob(args, probs_batch[b:b+1]).mean(1))

    with torch.no_grad():
        for image_batch, path in tqdm(in_loader):
            image_batch = image_batch.cuda()

            _ = model.encode_image(image_batch)
            stacked_features = process_intermediate_features(model, intermediate_features)
            image_features =  stacked_features.float()            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if "quickgelu" in args.models_name:
                logits = image_features @ text_features.t()

            if "biomed" in args.models_name:
                logits = model.logit_scale * image_features @ text_features.t()

            probs_batch = to_np(logits)            
            probs_batch = probs_batch[:, best_layers, :]

            for b in range(probs_batch.shape[0]):
                ind_score.append(get_scores_prob(args, probs_batch[b:b+1]).mean(1))

        # Remove hooks
    for hook in hooks:
        hook.remove()

    return ind_score, ood_score, best_layers



def get_ood_scores(args, in_images, ood_images, model, preprocess, tokenizer):
    ood_score = []
    ind_score = []
    
    if args.text_prompt == 'single-neg':

        in_class, neg_class = get_in_class_text(args)
        text = tokenizer([x for x in in_class]).cuda()
        text_neg = tokenizer([x for x in neg_class]).cuda()
        
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = model.encode_text(text_neg)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
    else:
        in_class = get_in_class_text(args)
        text = tokenizer([x for x in in_class]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    batch_size = 64   # or 8
    num_workers = 16   # 
    
    if 'oasis' in args.dataset_dir:
        crop_dim = 224
        preprocess = trn.Compose([tio.ToCanonical(), ToTensor(), AddChannelDim(), CropForeground(select_fn=empty_threshold, margin=0,  k_divisible=[crop_dim, crop_dim, crop_dim]), tio.CropOrPad((crop_dim,crop_dim,crop_dim)), ToFloat(), tio.ZNormalization()])
    
    in_dataset = ImagePathDataset(in_images, preprocess, args.dataset_dir)
    ood_dataset = ImagePathDataset(ood_images, preprocess, args.dataset_dir)

    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
        for image_batch, _ in tqdm(in_loader):
            image_batch = image_batch.cuda()
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if "quickgelu" in args.models_name:
                logits = image_features @ text_features.t()
                if args.text_prompt == 'single-neg':
                    logits_neg = image_features @ text_features_neg.t()
                    logits_neg = 1.0 - logits_neg.softmax(dim=-1).cpu().numpy()


            if "biomed" in args.models_name:
                logits = model.logit_scale * image_features @ text_features.t()
                if args.text_prompt == 'single-neg':
                    logits_neg = model.logit_scale * image_features @ text_features_neg.t()
                    logits_neg = 1.0 - logits_neg.softmax(dim=-1).cpu().numpy()           
            
            #import pdb; pdb.set_trace()
            if args.text_prompt == 'single-neg':
                probs_batch = (logits_neg + logits.softmax(dim=-1).cpu().numpy())/2.0
            else:
                probs_batch = logits.softmax(dim=-1).cpu().numpy()

            for b in range(probs_batch.shape[0]):
                ind_score.append(get_scores_prob(args, probs_batch[b:b+1]))

    with torch.no_grad():
        for image_batch, _ in tqdm(ood_loader):
            image_batch = image_batch.cuda()                
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if "quickgelu" in args.models_name:
                logits = image_features @ text_features.t()
                if args.text_prompt == 'single-neg':
                    logits_neg = image_features @ text_features_neg.t()
                    logits_neg = logits_neg.softmax(dim=-1).cpu().numpy()


            if "biomed" in args.models_name:
                logits = model.logit_scale * image_features @ text_features.t()
                if args.text_prompt == 'single-neg':
                    logits_neg = model.logit_scale * image_features @ text_features_neg.t()
                    logits_neg = logits_neg.softmax(dim=-1).cpu().numpy()           


            if args.text_prompt == 'single-neg':
                probs_batch = (logits_neg + logits.softmax(dim=-1).cpu().numpy())/2.0
            else:
                probs_batch = logits.softmax(dim=-1).cpu().numpy()

            for b in range(probs_batch.shape[0]):
                ood_score.append(get_scores_prob(args, probs_batch[b:b+1]))

    return ind_score, ood_score

def get_oasis_dataset(args):
    
    in_images = []
    ood_images = []

    in_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/train_oasis3.txt')
    
    for line in open(in_dir_path, encoding="utf-8"):
        if os.path.isfile(os.path.join(args.dataset_root_dir,'OpenMIBOOD/oasis',line.rstrip().split(' ')[0])):
            in_images.extend([os.path.join(args.dataset_root_dir,'OpenMIBOOD/oasis',line.rstrip().split(' ')[0])])     

    if args.dataset_dir == 'oasis-near-atlas':
        ood_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/test_oasis3_atlas_near.txt')
    if args.dataset_dir == 'oasis-near-brats':
        ood_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/test_oasis3_brats_near.txt') 
    if args.dataset_dir == 'oasis-near-ct':
        ood_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/test_oasis3_ct_near.txt') 
    if args.dataset_dir == 'oasis-far-heart':
        ood_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/test_oasis3_heart_far.txt')  
    if args.dataset_dir == 'oasis-far-chaos':
        ood_dir_path = os.path.join(args.dataset_root_dir, 'OpenMIBOOD/benchmark_imglist/oasis3/test_oasis3_chaos_inPhase_far.txt')   

    for line in open(ood_dir_path, encoding="utf-8"):
        if os.path.isfile(os.path.join(args.dataset_root_dir,'OpenMIBOOD/oasis',line.rstrip().split(' ')[0])):
            ood_images.extend([os.path.join(args.dataset_root_dir,'OpenMIBOOD/oasis',line.rstrip().split(' ')[0])])

    return in_images, ood_images
    

def set_model_dataset(args):
    in_images = []
    ood_images = []

    if args.dataset_dir == 'xray':
        in_dir =  'xray/normal-pneumonia'
        ood_dir = 'xray/covid'

    if args.dataset_dir == 'midog-csid':
        in_dir = 'OpenMIBOOD/midog/1a'
        ood_dir = 'OpenMIBOOD/midog/csid'

    if args.dataset_dir == 'midog-near':
        in_dir = 'OpenMIBOOD/midog/1a'
        ood_dir = 'OpenMIBOOD/midog/near'

    if args.dataset_dir == 'midog-far-fnac':
        in_dir = 'OpenMIBOOD/midog/1a'
        ood_dir = 'OpenMIBOOD/midog/far/fnac2019_crops'

    if args.dataset_dir == 'midog-far-ccagt':
        in_dir = 'OpenMIBOOD/midog/1a'
        ood_dir = 'OpenMIBOOD/midog/far/ccagt_crops'

    if 'oasis' in args.dataset_dir:
        in_images, ood_images = get_oasis_dataset(args)
        return in_images, ood_images

    in_dir_path = os.path.join(args.dataset_root_dir, in_dir)

    for root, _ , files in os.walk(in_dir_path):        
        in_images.extend([os.path.join(root, file) for file in  files])

    ood_dir_path = os.path.join(args.dataset_root_dir, ood_dir)

    for root, _ , files in os.walk(ood_dir_path):
        ood_images.extend([os.path.join(root, file) for file in  files])
    
    return in_images, ood_images

def set_model_clip(args):
    
    ckpt_mapping = {"ViT-B-16-quickgelu":"unimed_clip_vit_b16.pt"}

    
    
    if "quickgelu" in args.models_name:
        args.ckpt = ckpt_mapping[args.models_name]
        from src.open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
        mean, std = get_mean_std()
        text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        pretrained_weights = os.path.join(args.models_ckpt_dir, args.ckpt) 
        device='cuda'
        model, _, preprocess = create_model_and_transforms(args.models_name, pretrained_weights, precision='amp', device=device, force_quick_gelu=True, mean=mean, std=std, inmem=True, text_encoder_name=text_encoder_name,)
        tokenizer = HFTokenizer(text_encoder_name,context_length=256,**{},)

    elif "biomed" in args.models_name:
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    else:
        print('none models found !!')

    model.cuda()
    model.eval()
    return model, preprocess, tokenizer

