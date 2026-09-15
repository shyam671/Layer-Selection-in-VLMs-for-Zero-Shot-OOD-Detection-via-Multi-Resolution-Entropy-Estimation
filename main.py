import torch
import argparse
import numpy as np
from utils import setup_seed, set_model_clip, set_model_dataset, get_ood_scores, get_mod_scores

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def process_args():

    parser = argparse.ArgumentParser(description='MedOOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', default='xray', type=str,choices=['xray', 'midog-csid', 'midog-near', 'midog-far-fnac', 'midog-far-ccagt', 'oasis-near-atlas', 'oasis-near-brats', 'oasis-near-ct', 'oasis-far-heart', 'oasis-far-chaos'], help='in-ood-distribution dataset')
    parser.add_argument('--dataset_root_dir', default="/data/local/MedOOD/", type=str,help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood", type=str, help="unique ID for the run")
    parser.add_argument('--gpu', default=0, type = int, help='the GPU indice to use')
    parser.add_argument('--seed', default=0, type=int, help="random seed")

    parser.add_argument('--number_of_mod_layers', default=5, type=int, help="")
    parser.add_argument("--manual_input_best_layers", action="store_true")
    parser.add_argument('--input_best_layers', default=11, type=int, help="")


    parser.add_argument('--models_ckpt_dir', type=str, default='/data/local/MedOOD/models/',  help='where all the pretrained clip models are stored')    
    parser.add_argument('--models_name', type=str, default='ViT-B-16-quickgelu', choices=['ViT-B-16-quickgelu', 'biomed'], help='name of the clip model to be used')    
    parser.add_argument('--score', default='MCM', type=str, choices=['MCM', 'energy', 'entropy', 'var', 'MOD'], help='score options')
    parser.add_argument('--text_prompt', type=str, choices=['single', 'multi',  'multi-mod', 'single-neg'], help='score options')
    
    args = parser.parse_args()

    return args

def main():
    
    args = process_args()
    setup_seed(args.seed)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    model, preprocess, tokenizer = set_model_clip(args)

    in_images, ood_images = set_model_dataset(args)

    if args.score == 'MOD':
        ind_score, ood_score, best_layers = get_mod_scores(args, in_images, ood_images, model, preprocess, tokenizer)
    else:        
        ind_score, ood_score = get_ood_scores(args, in_images, ood_images, model, preprocess, tokenizer)

    concat_score = [x for x in ind_score] + [x for x in ood_score]
    concat_gt = [0 for x in range(len(ind_score))] + [1 for x in range(len(ood_score))]

    fpr, tpr, _ = roc_curve(concat_gt, concat_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(concat_gt, concat_score)
    aupr = average_precision_score(concat_gt, concat_score)

    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    if args.score == 'MOD':
        line = f"{args.dataset_dir:<20} {roc_auc*100:>6.1f} {aupr*100:>6.1f} {fpr95*100:>6.1f} {str(best_layers):<15}\n"
    else:
        line = f"{args.dataset_dir:<20} {roc_auc*100:>6.1f} {aupr*100:>6.1f} {fpr95*100:>6.1f}\n"

    with open("results.txt", "a") as f:
        f.write(line)


if __name__ == '__main__':
    main()