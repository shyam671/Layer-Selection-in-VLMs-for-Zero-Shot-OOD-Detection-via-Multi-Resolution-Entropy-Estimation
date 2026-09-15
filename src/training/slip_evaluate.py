# Copyright (c) Meta Platforms, Inc. and affiliates

import math
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None
import json
import os
from constants import CHEXPERT_CLASS_PROMPTS, CHEXPERT_CLASS_PROMPTS_webdataset, RSNA_CLASS_PROMPTS_webdataset, \
    RSNA_CLASS_PROMPTS, thyroid_us_prompts, breast_us_prompts, meniscus_mri_prompts, acl_mri_prompts, \
    radimagenet_all_prompts, ct_scan_labels, diabetic_retinopathy_prompts, PCAM, LC25000_lung, \
    LC25000_colon, NCK_CRC_prompts, BACH_prompts, Osteo_prompts, skin_cancer_prompts, \
    skin_tumor_prompts, SICAPv2_prompts, refuge_prompts, five_prompts, odir_retina_prompts

from tqdm import tqdm
from collections import defaultdict


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


SKIP_DATASETS = ["imagenet", "radimagenet", 'SICAPv2']
# SKIP_DATASETS = ["radimagenet", "imagenet", "rsna_pneumonia", "meniscal_mri", "breast_us", "acl_mri", "thyroid_us", 'PCAM', 'LC25000_lung', 'LC25000_colon', 'CT_axial', 'CT_coronal', 'CT_sagittal', 'dr_uwf', 'dr_regular', 'SICAPv2', 'NCK_CRC', 'skin_tumor', 'skin_cancer', 'BACH'] # 'BACH'


@torch.no_grad()
def slip_evaluate(args, model, val_transform, tokenizer, epoch=0):
    metrics = {}
    if not is_master(args):
        return metrics
    from clipeval import datasets, eval_zeroshot

    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")

    if hasattr(model, "module"):
        model = model.module

    for d in catalog:
        if d in SKIP_DATASETS:
            continue
        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        if d == 'chexpert-5x200':
            # templates = CHEXPERT_CLASS_PROMPTS
            templates = CHEXPERT_CLASS_PROMPTS_webdataset
            labels = None
        elif d == 'rsna_pneumonia':
            # templates = RSNA_CLASS_PROMPTS
            templates = RSNA_CLASS_PROMPTS_webdataset
            labels = None
        elif d == 'thyroid_us':
            templates = thyroid_us_prompts
            labels = None
        elif d == 'breast_us':
            templates = breast_us_prompts
            labels = None
        elif d == 'meniscal_mri':
            templates = meniscus_mri_prompts
            labels = None
        elif d == 'acl_mri':
            templates = acl_mri_prompts
            labels = None
        elif d == 'radimagenet':
            templates = radimagenet_all_prompts
            labels = None
        elif (d == "CT_axial") or (d == "CT_coronal") or (d == "CT_sagittal"):
            templates = ct_scan_labels
            labels = None
        elif (d == "dr_regular") or (d == "dr_uwf"):
            templates = diabetic_retinopathy_prompts
            labels = None
        elif d == 'LC25000_lung':
            templates = LC25000_lung
            labels = None
        elif d == 'LC25000_colon':
            templates = LC25000_colon
            labels = None
        elif d == 'PCAM':
            templates = PCAM
            labels = None
        elif d == 'NCK_CRC':
            templates = NCK_CRC_prompts
        elif d == 'BACH':
            templates = BACH_prompts
        elif d == 'Osteo':
            templates = Osteo_prompts
        elif d == 'skin_cancer':
            templates = skin_cancer_prompts
        elif d == 'SICAPv2':
            templates = SICAPv2_prompts
        elif d == 'skin_tumor':
            templates = skin_tumor_prompts
        elif d == 'refuge_retina':
            templates = refuge_prompts
        elif d == 'five_retina':
            templates = five_prompts
        elif d == 'odir_retina':
            templates = odir_retina_prompts
        else:
            templates = all_templates[d]
            labels = all_labels[d]

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size // 2, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False)

        metric = eval_zeroshot.evaluate(d, val_loader, templates, labels, model, tokenizer)
        metrics[d] = metric
        json_str = json.dumps({"task": d, "acc": metric})
        if args.rank == 0:
            print(json_str)
            with open(os.path.join(args.output_dir, "slip.txt"), mode="a+", encoding="utf-8") as f:
                f.write(f"Saving results for Epoch {epoch}" + "\n")
                f.write(json_str + "\n")
        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                if name == 'radimagenet':
                    wandb.log({f"val/{name}": val["acc"], 'epoch': epoch})
                elif name == 'chexpert-5x200' or name == 'chexpert-5x200' or name == 'CT_sagittal' or \
                        name == 'CT_axial' or name == 'CT_coronal' or name == 'dr_uwf' or name == 'dr_regular' or \
                        name == 'PCAM' or name == 'LC25000_lung' or name == 'LC25000_colon' \
                        or name == 'NCK_CRC' or name == 'BACH' or name == 'Osteo' \
                        or name == 'skin_cancer' or name == "skin_tumor" or name == 'SICAPv2' \
                        or name == 'five_retina' or name == 'odir_retina':

                    wandb.log({f"val/{name}": val, 'epoch': epoch})
                else:
                    wandb.log({f"val/{name}/acc": val['acc'],
                               f"val/{name}/auc_roc": val['auc_roc'],
                               f"val/{name}/precision_score": val['precision_score'],
                               f"val/{name}/f1_score": val['f1_score'],
                               f"val/{name}/recall_score": val['recall_score'],
                               'epoch': epoch})
    return metrics
