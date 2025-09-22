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

    # model
    model = build_model(cfg, num_classes=cfg.MODEL.NUM_CLASSES if hasattr(cfg.MODEL,'NUM_CLASSES') else 1000)
    sd = torch.load(args.weights, map_location='cpu')
    if isinstance(sd, dict) and 'module' in str(type(sd)):  # safe-guard
        sd = sd.module.state_dict()
    if 'state_dict' in sd:  # handle ignite checkpoints
        sd = sd['state_dict']
    try:
        model.load_state_dict(sd if isinstance(sd, dict) else sd.module.state_dict(), strict=False)
    except:
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd.module.state_dict(), strict=False)

    model.eval().to(device)
    if hasattr(model, 'module'): m = model.module
    else: m = model

    # transforms: reuse test-time transforms
    from data.transforms import build_transforms
    tfm = build_transforms(cfg, is_train=False)

    # read pairs
    pairs = []
    with open(args.pairs_file, 'r') as f:
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
