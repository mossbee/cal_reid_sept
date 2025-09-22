import argparse, os, sys, torch, numpy as np
from PIL import Image
from torchvision import transforms
from sklearn import metrics

sys.path.append('.')
from config import cfg
from modeling import build_model

def load_img(path, tfm):
    img = Image.open(path).convert('RGB')
    return tfm(img).unsqueeze(0)

def cosine_similarity(a, b):
    return (a * b).sum(-1)

def eer(fpr, tpr):
    # Find point where FPR ~= 1-TPR
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return max(fpr[idx], fnr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', type=str, default='')
    ap.add_argument('--pairs_file', type=str, required=True, help='txt: imgA imgB label(0/1)')
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # model - try to infer num_classes from the checkpoint first
    num_classes = 1000  # default
    if hasattr(cfg.MODEL, 'NUM_CLASSES'):
        num_classes = cfg.MODEL.NUM_CLASSES
    else:
        # Try to infer from the checkpoint
        try:
            temp_sd = torch.load(args.weights, map_location='cpu', weights_only=False)
            if hasattr(temp_sd, 'module'):
                temp_sd = temp_sd.module.state_dict()
            elif isinstance(temp_sd, dict) and 'state_dict' in temp_sd:
                temp_sd = temp_sd['state_dict']
            elif isinstance(temp_sd, dict) and 'module' in temp_sd:
                temp_sd = temp_sd['module']
            
            # Look for classifier weight to infer num_classes
            for key in temp_sd.keys():
                if 'classifier.weight' in key:
                    num_classes = temp_sd[key].shape[0]
                    print(f"Inferred num_classes from checkpoint: {num_classes}")
                    break
        except Exception as e:
            print(f"Could not infer num_classes from checkpoint, using default: {num_classes}")
    
    model = build_model(cfg, num_classes=num_classes)
    sd = torch.load(args.weights, map_location='cpu', weights_only=False)
    
    # Handle different types of saved models
    if hasattr(sd, 'module'):  # DataParallel model
        sd = sd.module.state_dict()
    elif isinstance(sd, dict):
        if 'state_dict' in sd:  # handle ignite checkpoints
            sd = sd['state_dict']
        elif 'module' in sd:  # another DataParallel format
            sd = sd['module']
    else:
        # Assume it's already a state dict
        pass
    
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"Warning: Failed to load state dict with strict=False: {e}")
        # Try loading with even more relaxed constraints
        try:
            # Extract only the matching keys
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in sd.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} matching parameters")
        except Exception as e2:
            print(f"Error: Could not load any parameters: {e2}")
            raise

    model.eval().to(device)
    if hasattr(model, 'module'): m = model.module
    else: m = model

    # transforms: reuse test-time transforms
    from data.transforms import build_transforms
    tfm = build_transforms(cfg, is_train=False)

    # read pairs
    pairs = []
    try:
        with open(args.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                a, b, y = line.split()
                pairs.append((a, b, int(y)))
    except UnicodeDecodeError:
        # Try with different encodings if UTF-8 fails
        try:
            with open(args.pairs_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    a, b, y = line.split()
                    pairs.append((a, b, int(y)))
        except UnicodeDecodeError:
            with open(args.pairs_file, 'r', encoding='cp1252') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    a, b, y = line.split()
                    pairs.append((a, b, int(y)))

    # cache features per image
    paths = sorted(set([p for a,b,_ in pairs for p in (a,b)]))
    feats = {}
    with torch.no_grad():
        batch_imgs, batch_paths = [], []
        for p in paths:
            img = load_img(p, tfm)
            batch_imgs.append(img)
            batch_paths.append(p)
            if len(batch_imgs) == args.batch_size:
                x = torch.cat(batch_imgs, 0).to(device)
                f = m.extract_feat(x).cpu().numpy()
                for bp, bf in zip(batch_paths, f):
                    feats[bp] = bf
                batch_imgs, batch_paths = [], []
        if batch_imgs:
            x = torch.cat(batch_imgs, 0).to(device)
            f = m.extract_feat(x).cpu().numpy()
            for bp, bf in zip(batch_paths, f):
                feats[bp] = bf

    # compute similarities and metrics
    scores, labels = [], []
    for a, b, y in pairs:
        fa = feats[a]; fb = feats[b]
        scores.append(float(np.dot(fa, fb)))  # cosine since L2-normalized
        labels.append(y)
    scores = np.array(scores); labels = np.array(labels)

    # ROC / AUC
    fpr, tpr, thr = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    eer_val = eer(fpr, tpr)

    # Accuracy at best threshold
    best_thr_idx = np.argmax(tpr - fpr)
    thr_star = thr[best_thr_idx]
    preds = (scores >= thr_star).astype(int)
    acc = (preds == labels).mean()

    print(f'Verification ACC: {acc:.4f}')
    print(f'AUC ROC: {auc:.4f}')
    print(f'EER: {eer_val:.4f}')
    print(f'Best threshold: {thr_star:.4f}')

if __name__ == '__main__':
    main()
