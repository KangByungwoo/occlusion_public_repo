import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
import torchvision.transforms as T 
import torch.nn as nn
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multi_object_classifier import *

import os
import random
import math
import torch.nn.functional as F

import copy
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'            # Keep text as text (editable) in the SVG.
mpl.rcParams['font.family'] = 'sans-serif'         # Set font family.
# mpl.rcParams['font.sans-serif'] = ['Arial']          # Use Arial as the sans-serif font.
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Alternative to Arial

mpl.rcParams['font.size'] = 12                     # Set the font size to 12.


# Set the seed for PyTorch
torch.manual_seed(42)

# Set the seed for CUDA (if you're using GPUs)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If you use multi-GPU

# Set the seed for Python's random module
random.seed(42)

# Set the seed for NumPy
np.random.seed(42)

# Optional: Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def refine_mask(img_50, eps=1e-2, eps2=1e-1):
    """
    Refine the mask just like in your original code:
      1) rough binary mask thresholding by eps
      2) compute average intensity in that mask
      3) threshold again by eps2 * avg_intensity
    Returns: refined_mask of shape (1,50,50)
    """
    mask = (img_50.mean(dim=0, keepdim=True) > eps)  # shape (1,50,50)

    masked_sum = (img_50 * mask.float()).view(-1).sum()
    masked_area = mask.float().view(-1).sum().clamp(min=1.0)  # avoid div by 0
    avg_intensity = masked_sum / masked_area

    refined_mask = (img_50.mean(dim=0, keepdim=True) > eps2 * avg_intensity)
    return refined_mask

def apply_transform(
    single_28,  # shape (1,28,28)
    angle,      # in radians
    trans=(0.0,0.0),
    eps=1e-2, eps2=1e-1
):
    """
    1) Pad 1x28x28 to 1x50x50
    2) Rotate by 'angle', translate by (0,0) (or your chosen fixed trans)
    3) Refine mask
    Returns: (img_out, mask_out), each shape (1,50,50)
    """
    # 1) Pad to 1×50×50. We'll center-pad by 11 on each side to reach 50×50.
    img_28 = single_28  # shape (1,28,28)
    img_50 = F.pad(img_28, (11,11,11,11))  # shape (1,50,50)
    
    # 2) Build the affine matrix
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # trans are in normalized coords for affine_grid (range [-1..1]).
    # If you want a pixel shift, you need to scale by (trans_x / (width/2)).
    # But you're keeping it (0,0), so no shift anyway.
    affine_mat = torch.tensor(
        [[ cos_a, -sin_a, trans[0]],
         [ sin_a,  cos_a, trans[1]]],
        dtype=torch.float
    ).unsqueeze(0)  # shape (1,2,3)

    # Make it shape (N,C,H,W)= (1,1,50,50)
    img_50 = img_50.unsqueeze(0)  # shape (1,1,50,50)

    # Create the sampling grid
    grid = F.affine_grid(affine_mat, img_50.shape, align_corners=False)
    img_rot = F.grid_sample(img_50, grid, align_corners=False)  # (1,1,50,50)

    # Squeeze out the batch dimension => shape (1,50,50)
    img_rot_1x50x50 = img_rot.squeeze(0)  # shape (1,50,50)

    # 3) Refine mask
    mask_1x50x50 = refine_mask(img_rot_1x50x50, eps=eps, eps2=eps2)

    return img_rot_1x50x50, mask_1x50x50



def compose_two_objects(bg_img, bg_mask, fg_img, fg_mask):
    """
    Combine background & foreground:
      out = fg_mask * fg_img + (1 - fg_mask) * bg_img
    Each input is shape (1,50,50).
    Returns shape (1,50,50).
    """
    composed = fg_mask.float() * fg_img + (1 - fg_mask.float()) * bg_img
    return composed











import random
import torch
import os
import math
import torch.nn.functional as F


def load_single_object_random(category, reduced_data_dir='./reduced_fashion_mnist/0289/test'):
    """
    Loads a random single-object 28×28 image from the specified 'category' folder/file.
    Returns:
       single_28  : shape (1,28,28)  [FloatTensor in 0..1]
       index_used : the integer index of the chosen image in that category's dataset
    """
    data = torch.load(os.path.join(reduced_data_dir, 'data.pt'), map_location=torch.device('cpu'))  # shape (N,1,28,28), for example
    labels = torch.load(os.path.join(reduced_data_dir, 'labels.pt'), map_location=torch.device('cpu'))  # shape (N,), for example
    cat_idxs = torch.nonzero(labels == category)[:,0] # torch.nonzero(labels == category) has shape (N,1), so we remove the col dim.
    cat_data = data[cat_idxs]

    idx = random.randint(0, len(cat_data)-1)
    single_28 = cat_data[idx]   # shape (1,28,28)
    return single_28, cat_idxs[idx]



def load_single_object_by_index(category, index, reduced_data_dir='./reduced_fashion_mnist/0289/test'):
    """
    Loads the EXACT single-object 28×28 at 'index' from 'category'.
    Returns (1,28,28).
    """
    data = torch.load(os.path.join(reduced_data_dir, 'data.pt'), map_location=torch.device('cpu'))  # shape (N,1,28,28), for example
    single_28 = data[index]  # shape (1,28,28)
    return single_28







def generate_single_x1(
    valid_classes,
    reduced_data_dir='./reduced_fashion_mnist/0289/test',
    angle_min=-math.pi, angle_max=math.pi,
    eps=1e-2, eps2=1e-1
):
    """
    1) Random background => (bg_cat, bg_idx)
    2) Random foreground => (fg_cat, fg_idx)
    3) Random angles => compose x1
    Returns x1, meta1
       x1: shape (1,50,50)
       meta1: dict with:
         - bg_cat, bg_idx, bg_angle
         - fg_cat, fg_idx, fg_angle
    """

    trans_angle = random.uniform(0, 2 * math.pi)
    trans_mag = random.uniform(0.1, 0.15)
    trans = trans_mag * torch.tensor([math.cos(trans_angle), math.sin(trans_angle)])
    bg_trans = trans
    fg_trans = -trans

    # Random background
    bg_cat = random.choice(valid_classes)
    bg_28, bg_idx = load_single_object_random(bg_cat, reduced_data_dir)
    bg_angle = random.uniform(angle_min, angle_max)
    bg_img, bg_mask = apply_transform(bg_28, bg_angle, bg_trans, eps, eps2)

    # Random foreground
    fg_cat = random.choice(valid_classes)
    fg_28, fg_idx = load_single_object_random(fg_cat, reduced_data_dir)
    fg_angle = random.uniform(angle_min, angle_max)
    fg_img, fg_mask = apply_transform(fg_28, fg_angle, fg_trans, eps, eps2)

    # Compose
    x1 = compose_two_objects(bg_img, bg_mask, fg_img, fg_mask)

    # Compute visible ratio VR for n_objs=2:
    # intersection of both object masks:
    x1_intsec = bg_mask * fg_mask

    N = x1_intsec.size(0)
    x1_vr = 1.0 - x1_intsec.float().view(N, -1).sum(1) / \
                bg_mask.float().view(N, -1).sum(1)



    # meta
    meta1 = {
        'bg_cat': bg_cat,
        'bg_idx': bg_idx,      # EXACT background instance
        'bg_angle': bg_angle,
        'bg_trans': bg_trans,

        'fg_cat': fg_cat,
        'fg_idx': fg_idx,      # EXACT foreground instance
        'fg_angle': fg_angle,
        'fg_trans': fg_trans,
        'x1_vr': x1_vr
    }
    return x1, meta1



def load_exact_foreground(fg_cat, fg_idx, fg_angle, fg_trans, reduced_data_dir, eps, eps2):
    """
    Reload EXACT same foreground single-object 28×28 from (fg_cat, fg_idx),
    apply angle=fg_angle, return (fg_img, fg_mask).
    """
    fg_28 = load_single_object_by_index(fg_cat, fg_idx, reduced_data_dir)  # EXACT
    fg_img, fg_mask = apply_transform(fg_28, fg_angle, fg_trans, eps, eps2)
    return fg_img, fg_mask

def generate_x2_for_scenario(
    scenario,
    x1_meta,
    valid_classes,
    reduced_data_dir='./reduced_fashion_mnist/0289/test',
    angle_min=-math.pi, angle_max=math.pi,
    eps=1e-2, eps2=1e-1
):
    """
    Generate x2 by preserving the foreground from x1 but altering 
    the background based on the scenario ('category', 'orientation', or 'both').
    Returns x2, meta2.
    """
    # Preserve EXACT foreground from x1
    fg_cat = x1_meta['fg_cat']
    fg_idx = x1_meta['fg_idx']
    fg_angle = x1_meta['fg_angle']
    fg_trans = x1_meta['fg_trans']
    x1_vr = x1_meta['x1_vr']

    fg_img, fg_mask = load_exact_foreground(fg_cat, fg_idx, fg_angle, fg_trans,
                                           reduced_data_dir, eps, eps2)

    # Retrieve original background parameters from x1
    bg_cat1 = x1_meta['bg_cat']
    bg_idx1 = x1_meta['bg_idx']
    bg_angle1 = x1_meta['bg_angle']
    bg_trans1 = x1_meta['bg_trans']

    # Determine new background attributes based on the scenario
    if scenario == 'category':
        # Preserve orientation, change category
        angle2 = bg_angle1
        bg_cat2 = random.choice([c for c in valid_classes if c != bg_cat1])
        bg_28_2, bg_idx2 = load_single_object_random(bg_cat2, reduced_data_dir)
        bg_trans2 = bg_trans1

    elif scenario == 'orientation':
        # Preserve category and instance, change orientation
        bg_cat2 = bg_cat1
        bg_idx2 = bg_idx1
        while True:
            candidate_angle = random.uniform(angle_min, angle_max)
            if abs(candidate_angle - bg_angle1) > math.pi/180*30:  # at least 30° difference
                angle2 = candidate_angle
                break
        bg_trans2 = bg_trans1

    else:  # scenario == 'both'
        # Change both category and orientation
        bg_cat2 = random.choice([c for c in valid_classes if c != bg_cat1])
        bg_28_2, bg_idx2 = load_single_object_random(bg_cat2, reduced_data_dir)
        while True:
            candidate_angle = random.uniform(angle_min, angle_max)
            if abs(candidate_angle - bg_angle1) > math.pi/180*30:
                angle2 = candidate_angle
                break
        bg_trans2 = bg_trans1

    # For 'orientation' scenario, reload the exact same background instance before transforming
    if scenario == 'orientation':
        bg_28_2 = load_single_object_by_index(bg_cat2, bg_idx2, reduced_data_dir)
    # For other scenarios, bg_28_2 was already loaded

    # Apply transformation to the new background
    bg_img2, bg_mask2 = apply_transform(bg_28_2, angle2, bg_trans2, eps, eps2)

    # Compose the new image with the new background and preserved foreground
    x2 = compose_two_objects(bg_img2, bg_mask2, fg_img, fg_mask)

    # Compute visible ratio VR for the composed image
    x2_intsec = bg_mask2 * fg_mask
    N = x2_intsec.size(0)
    x2_vr = 1.0 - x2_intsec.float().view(N, -1).sum(1) / \
                bg_mask2.float().view(N, -1).sum(1)

    # Construct meta information for x2
    meta2 = {
        'bg_cat1': bg_cat1,
        'bg_idx1': bg_idx1,
        'bg_angle1': bg_angle1,
        'bg_trans1': bg_trans1,
        'fg_cat': fg_cat,
        'fg_idx': fg_idx,
        'fg_angle': fg_angle,
        'fg_trans': fg_trans,
        'bg_cat2': bg_cat2,
        'bg_idx2': bg_idx2,
        'bg_angle2': angle2,
        'bg_trans2': bg_trans2,
        'x1_vr': x1_vr,
        'x2_vr': x2_vr,
        'scenario': scenario
    }
    return x2, meta2



def generate_pairs_same_x1_exact_instances(
    valid_classes,
    n_x1=1000,
    out_dir='./pair_data',
    reduced_data_dir='./reduced_fashion_mnist/0289/test',
    angle_min=-math.pi,
    angle_max=math.pi,
    eps=1e-2,
    eps2=1e-1
):
    """
    1) Generate n_x1 distinct x1 images, each with a 
       specific background instance (bg_cat,bg_idx) 
       and a specific foreground instance (fg_cat,fg_idx).
    2) For each scenario in ['category','orientation','both'],
       build x2 that modifies the FG instance accordingly, 
       but uses EXACT same background instance index.
    3) Save them in out_dir/<scenario>/pairs.pt
       The same x1 is used in all three scenario sets, 
       but x2 changes.
    """
    # Save
    all_x1_save_path = os.path.join(out_dir, 'all_x1.pt')
    if os.path.isfile(all_x1_save_path):
        print(f"Dataset already exists at {all_x1_save_path}. Loading...")
        all_x1 = torch.load(all_x1_save_path)
    else:
        # Step A: gather x1 images
        all_x1 = []
        for _ in range(n_x1):
            x1, meta1 = generate_single_x1(
                valid_classes,
                reduced_data_dir,
                angle_min, angle_max,
                eps, eps2
            )
            all_x1.append((x1, meta1))
        
        torch.save(all_x1, all_x1_save_path)
        print(f"Saved all_x1 to {all_x1_save_path}")


    # Step B: produce 3 scenario subfolders
    scenario_list = ['category', 'orientation', 'both']
    for scenario in scenario_list:
        scenario_dir = os.path.join(out_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)

        save_path = os.path.join(scenario_dir, 'pairs.pt')
        if os.path.isfile(save_path):
            print(f"Dataset for scenario='{scenario}' already exists at {save_path}. Skipping...")
            continue

        all_pairs = []
        for (x1, meta1) in all_x1:
            # Create x2
            x2, meta2 = generate_x2_for_scenario(
                scenario, meta1,
                valid_classes,
                reduced_data_dir,
                angle_min, angle_max,
                eps, eps2
            )
            # Combine into final dict
            pair_dict = {
                'x1': x1,
                'x2': x2,
                'x1_vr':meta2['x1_vr'],
                'x2_vr':meta2['x2_vr'],
                # store the final scenario metadata
                'bg_cat': meta2['bg_cat'],
                'bg_idx': meta2['bg_idx'],
                'bg_angle': meta2['bg_angle'],
                'bg_trans': meta2['bg_trans'],

                'fg_cat1': meta2['fg_cat1'],
                'fg_idx1': meta2['fg_idx1'],
                'fg_angle1': meta2['fg_angle1'],
                'fg_trans1': meta2['fg_trans1'],

                'fg_cat2': meta2['fg_cat2'],
                'fg_idx2': meta2['fg_idx2'],
                'fg_angle2': meta2['fg_angle2'],
                'fg_trans2': meta2['fg_trans2'],

                'scenario': meta2['scenario']
            }
            all_pairs.append(pair_dict)

        # Save
        save_path = os.path.join(scenario_dir, 'pairs.pt')
        torch.save(all_pairs, save_path)
        print(f"Saved {len(all_pairs)} pairs for scenario='{scenario}' to {save_path}")





def main_generate_all(out_dir):
    valid_classes = [0, 1, 2, 3]  # example
    generate_pairs_same_x1_exact_instances(
        valid_classes=valid_classes,
        n_x1=5000,
        out_dir=out_dir,
        reduced_data_dir='./reduced_fashion_mnist/0289/test'
    )
    print("Done generating same-x1 exact-instance dataset.")



class OccludedPairDataset(Dataset):
    def __init__(self, pair_file):
        """
        pair_file: e.g. './pair_data/orientation/pairs.pt'
        We'll load them all into memory for convenience.
        """
        super().__init__()
        print(f"Loading pairs from {pair_file}...")
        self.pairs = torch.load(pair_file)  # a list of dicts
        print(f"Loaded {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns (x1, x2, meta_dict), 
        where x1, x2 are each (1,50,50) FloatTensors,
        and meta_dict includes background/foreground info (cat, angle, etc.).
        """
        sample = self.pairs[idx]
        x1 = sample['x1']
        x2 = sample['x2']
        fg_cat = sample['fg_cat']
        return x1, x2, fg_cat

def get_pair_dataloader(pair_file, batch_size=16, shuffle=False):
    dataset = OccludedPairDataset(pair_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_initial_states(model, batch_size):
    """
    Returns the initial states (h0, c0) for each of the three ConvLSTM layers
    and the final LSTM layer, expanded to batch_size.
    """
    convlstm1_hidden0 = (
        model.convlstm1_h0.expand(batch_size, -1, -1, -1),
        model.convlstm1_c0.expand(batch_size, -1, -1, -1)
    )
    convlstm2_hidden0 = (
        model.convlstm2_h0.expand(batch_size, -1, -1, -1),
        model.convlstm2_c0.expand(batch_size, -1, -1, -1)
    )
    convlstm3_hidden0 = (
        model.convlstm3_h0.expand(batch_size, -1, -1, -1),
        model.convlstm3_c0.expand(batch_size, -1, -1, -1)
    )
    lstm_hidden0 = (
        model.lstm_h0.expand(batch_size, -1),
        model.lstm_c0.expand(batch_size, -1)
    )
    return convlstm1_hidden0, convlstm2_hidden0, convlstm3_hidden0, lstm_hidden0






def test_two_pass_experiment(
    model, x1, x2, 
    device='cuda', 
    which_layers_to_replace=('convlstm1', 'convlstm2', 'convlstm3', 'lstm')
):
    """
    x1: original occluded image batch (N, 1, 50, 50)
    x2: modified occluded image batch (same background, different foreground/angle)
    model: a trained RCMultiObjectClassifier
    which_layers_to_replace: tuple or list of layer-names to overwrite in the second pass.

    Returns:
      - pass1_scores: (scores1_x1, scores2_x1) for the baseline pass
      - pass2_scores: (scores1_m1, scores2_m1) for the 'hidden-state-overwritten' pass
    """
    model.eval()
    x1 = x1.to(device)
    x2 = x2.to(device)
    batch_size = x1.size(0)

    # -------------------------
    # PASS 1: Baseline on x1->x1
    # -------------------------
    with torch.no_grad():
        # 1) Time-step-1 on x1
        h1_0, h2_0, h3_0, hlstm_0 = get_initial_states(model, batch_size)
        scores1_x1, h1_1, h2_1, h3_1, hlstm_1 = model.cell(
            predict=True, x=x1,
            convlstm1_hidden0=h1_0, 
            convlstm2_hidden0=h2_0, 
            convlstm3_hidden0=h3_0, 
            lstm_hidden0=hlstm_0
        )

        # 2) Time-step-2 on x1
        scores2_x1, h1_2, h2_2, h3_2, hlstm_2 = model.cell(
            predict=True, x=x1,
            convlstm1_hidden0=h1_1, 
            convlstm2_hidden0=h2_1, 
            convlstm3_hidden0=h3_1, 
            lstm_hidden0=hlstm_1
        )

    pass1_scores = (scores1_x1, scores2_x1)

    # ---------------------------------------------
    # PASS 2: Overwrite selected hidden states only
    # ---------------------------------------------
    model1 = copy.deepcopy(model).eval()  # copy of the trained model
    model2 = copy.deepcopy(model).eval()  # second copy

    with torch.no_grad():
        # (A) Time-step-1 with model1 on x1
        h1_0_m1, h2_0_m1, h3_0_m1, hlstm_0_m1 = get_initial_states(model1, batch_size)
        scores1_m1, h1_1_m1, h2_1_m1, h3_1_m1, hlstm_1_m1 = model1.cell(
            predict=True, x=x1,
            convlstm1_hidden0=h1_0_m1, 
            convlstm2_hidden0=h2_0_m1, 
            convlstm3_hidden0=h3_0_m1, 
            lstm_hidden0=hlstm_0_m1
        )

        # (B) Time-step-1 with model2 on x2
        h1_0_m2, h2_0_m2, h3_0_m2, hlstm_0_m2 = get_initial_states(model2, batch_size)
        scores1_m2, h1_1_m2, h2_1_m2, h3_1_m2, hlstm_1_m2 = model2.cell(
            predict=True, x=x2,
            convlstm1_hidden0=h1_0_m2, 
            convlstm2_hidden0=h2_0_m2, 
            convlstm3_hidden0=h3_0_m2, 
            lstm_hidden0=hlstm_0_m2
        )

        # (C) Overwrite only the selected hidden states in model1 with model2
        if 'convlstm1' in which_layers_to_replace:
            h1_1_m1 = h1_1_m2
        if 'convlstm2' in which_layers_to_replace:
            h2_1_m1 = h2_1_m2
        if 'convlstm3' in which_layers_to_replace:
            h3_1_m1 = h3_1_m2
        if 'lstm' in which_layers_to_replace:
            hlstm_1_m1 = hlstm_1_m2

        # (D) Time-step-2 in model1 with x1, but "tainted" hidden states from x2
        scores2_m1, h1_2_m1, h2_2_m1, h3_2_m1, hlstm_2_m1 = model1.cell(
            predict=True, x=x1,
            convlstm1_hidden0=h1_1_m1, 
            convlstm2_hidden0=h2_1_m1, 
            convlstm3_hidden0=h3_1_m1, 
            lstm_hidden0=hlstm_1_m1
        )

    pass2_scores = (scores1_m1, scores2_m1)
    return pass1_scores, pass2_scores


def evaluate_model_on_scenario_with_strategies(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test_loader for pass1 accuracy 
    and pass2 accuracy for each replace-strategy.
    Returns a dict:
      {
        'pass1': pass1_acc,
        'all': pass2_acc_when_replacing_all_layers,
        'clstm1_only': ...,
        'clstm2_only': ...,
        'clstm3_only': ...,
        'lstm_only': ...
      }
    """

    replace_strategies = {
        'all': ('convlstm1', 'convlstm2', 'convlstm3', 'lstm'),
        'clstm1_only': ('convlstm1',),
        'clstm2_only': ('convlstm2',),
        'clstm3_only': ('convlstm3',),
        'lstm_only':   ('lstm',)
    }

    total_samples = 0

    # We'll keep running sums of correct predictions
    pass1_correct_sum = 0
    pass2_correct_sum = {k: 0 for k in replace_strategies.keys()}

    for x1_batch, x2_batch, fg_cat_batch in test_loader:
        x1_batch, x2_batch, fg_cat_batch = x1_batch.to(device), x2_batch.to(device), fg_cat_batch.to(device)

        # (A) Get pass1 predictions (2 steps both with X1), ignoring hidden-state injection
        pass1_scores, _ = test_two_pass_experiment(
            model, x1_batch, x2_batch, 
            device=device,
            which_layers_to_replace=()  # no overwriting => baseline pass1
        )
        pass1_logits = pass1_scores[1]
        _, pass1_preds = torch.max(pass1_logits, dim=1)
        pass1_correct_sum += (pass1_preds == fg_cat_batch).sum().item()

        # (B) For each strategy, do pass2 injection and measure accuracy
        for strategy_name, layers_to_replace in replace_strategies.items():
            _, pass2_scores = test_two_pass_experiment(
                model, x1_batch, x2_batch, 
                device=device,
                which_layers_to_replace=layers_to_replace
            )
            pass2_logits = pass2_scores[1]
            _, pass2_preds = torch.max(pass2_logits, dim=1)

            pass2_correct_sum[strategy_name] += (pass2_preds == fg_cat_batch).sum().item()

        total_samples += fg_cat_batch.size(0)

    # Now compute final accuracies
    pass1_acc = pass1_correct_sum / total_samples
    pass2_acc = {}
    for strategy_name in replace_strategies.keys():
        pass2_acc[strategy_name] = pass2_correct_sum[strategy_name] / total_samples

    # Return a single dict
    results_dict = {
        'pass1': pass1_acc,
        **pass2_acc
    }
    return results_dict




def main_test_with_replacement_strategies(pair_data_dir, model_dir, results_file, plot_save_path):
    """
    Evaluate each model in `model_dir` across scenarios in `pair_data_dir`.
    For each scenario, compute pass1 accuracy and pass2 accuracy 
    for each of the 5 replacement strategies.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenarios = ['category', 'orientation', 'both']
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth.tar')]

    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}...")
        all_results = torch.load(results_file)
    else:
        all_results = {}

        for model_file in models:
            print(f"Evaluating model '{model_file}'...")
            model_path = os.path.join(model_dir, model_file)
            model = RCMultiObjectClassifier(reverse=True, n_classes=4, dropout_p=0)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            scenario_results = {}
            for scenario in scenarios:
                pair_file = os.path.join(pair_data_dir, scenario, 'pairs.pt')
                test_loader = get_pair_dataloader(pair_file, batch_size=1000, shuffle=False)

                # Evaluate pass1 + pass2 (with all strategies)
                scenario_results[scenario] = evaluate_model_on_scenario_with_strategies(
                    model, test_loader, device=device
                )
            
            all_results[model_file] = scenario_results
        
        # Save results
        torch.save(all_results, results_file)
        print(f"Results saved to {results_file}")

    plot_all_strategies_single_figure(
        all_results,
        scenarios=('category','orientation','both'),
        strategies=('all','clstm1_only','clstm2_only','clstm3_only','lstm_only'),
        save_path=plot_save_path
    )


def plot_all_strategies_single_figure(
    all_results,
    scenarios=('category','orientation','both'),
    strategies=('all','clstm1_only','clstm2_only','clstm3_only','lstm_only'),
    save_path='combined_strategies.png'
):
    """
    Creates ONE figure in which the x-axis = [pass1, category, orientation, both].
    For each strategy (color), we:
      - Plot a thick line + big dots for the average across all models
      - Plot thin lines + small dots for each individual model
    """
    fig, ax = plt.subplots(figsize=(4,3))
    
    # We'll define four x-positions:
    xvals = np.array([0,1,2,3])
    xlabels = ['pass1','category','orientation','both']

    # Some color palette for each strategy
    color_map = plt.get_cmap('tab10')  # or define your own list of colors
    model_names = list(all_results.keys())
    n_models = len(model_names)

    # 1) Compute a single "pass1" value per model by averaging pass1 across the 3 scenarios
    pass1_per_model = []
    for m in model_names:
        p1_vals = [all_results[m][sc]['pass1'] for sc in scenarios]
        pass1_per_model.append(np.mean(p1_vals))  # you could pick just one scenario if you prefer
    pass1_per_model = np.array(pass1_per_model)  # shape (n_models,)

    # 2) Loop over each replacement strategy
    for si, strategy in enumerate(strategies):
        # pick a color for this strategy
        c = color_map(si)

        # We'll collect data in shape (n_models, 4)
        #   data[:,0] = pass1 for each model
        #   data[:,1] = pass2 (category scenario) for each model
        #   data[:,2] = pass2 (orientation scenario) ...
        #   data[:,3] = pass2 (both scenario) ...
        data = np.zeros((n_models, 4))

        # fill column 0 with pass1
        data[:,0] = pass1_per_model

        # fill columns 1..3 with pass2 accuracies from all_results
        # for the respective scenario
        for i, sc in enumerate(scenarios, start=1):
            for m_idx, m_name in enumerate(model_names):
                data[m_idx,i] = all_results[m_name][sc][strategy]

        # 3) Compute the average across models
        avg_vals = data.mean(axis=0)

        # 4) Plot the average line with large markers
        ax.plot(
            xvals, 
            avg_vals, 
            '-o', 
            color=c, 
            linewidth=3, 
            markersize=10,
            label=strategy  # so we can see each strategy in the legend
        )

        # 5) Plot each model's line with partial alpha + smaller markers
        for m_idx in range(n_models):
            ax.plot(
                xvals,
                data[m_idx,:],
                '-o',
                color=c,
                alpha=0.3,      # partially transparent
                linewidth=1,
                markersize=5
            )

    # Some aesthetics
    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels)
    ax.set_ylim([0.7,0.95])
    ax.set_ylabel("Accuracy")
    ax.set_title("Combined Plot of Replacement Strategies")
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=300)
    plt.savefig(save_path + '.svg', format='svg')

    plt.close()








def plot_occ_level(pair_data_dir, plot_dir):
    for scenario in ['category', 'orientation', 'both']:
        plot_occ_level_for_scenario(pair_data_dir, scenario, plot_dir)

def plot_occ_level_for_scenario(pair_data_dir, scenario, plot_dir):

    plot_save_path = os.path.join(plot_dir, f'{scenario}_occ_level_plot.png')
    # Load the dataset
    pairs = torch.load(os.path.join(pair_data_dir, scenario, 'pairs.pt'))

    # Extract 1 - x1_vr and 1 - x2_vr for all pairs
    x1_vr_list = [1 - pair['x1_vr'].item() for pair in pairs]
    x2_vr_list = [1 - pair['x2_vr'].item() for pair in pairs]

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    plt.hist(x1_vr_list, bins=50, alpha=0.7, label='x1 occ level', color='blue', edgecolor='black')
    plt.hist(x2_vr_list, bins=50, alpha=0.7, label='x2 occ level', color='orange', edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of x1 and x2 occ level')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(plot_save_path, dpi=300)
    plt.close()



def main():
    pair_data_dir = './hidden_state_pert_test_for_multi_object_classifier_RCReversed_pair_data'
    results_dir = './hidden_state_pert_test_for_multi_object_classifier_each_layer_RCReversed_results'
    if not os.path.exists(pair_data_dir):
        os.makedirs(pair_data_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)


    model_dir = './multi_object_classifier_best_hp_test/extensive_info_multi_fashion_mnist_rotated_-180_180_trans_0.10_0.15_0289/2_28_50_0.25_0.50_1e-02_1e-01/RCReversed'
    
    results_file = os.path.join(results_dir, 'results.pth')
    plot_save_path = os.path.join(results_dir, 'results_plot')

    generate_data = False
    test = True
    do_plot_occ_level = False

    if generate_data:
        main_generate_all(pair_data_dir)
    
    if test:
        main_test_with_replacement_strategies(pair_data_dir, model_dir, results_file, plot_save_path)
    
    if do_plot_occ_level:
        plot_occ_level(pair_data_dir, results_dir)





if __name__ == '__main__':
    main()