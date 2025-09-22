import logging
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from data.transforms import build_transforms


def create_supervised_trainer(model, optimizer, loss_fn,using_cal,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)
        
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch

        img = img.cuda()
        target = target.cuda()
        if using_cal:
            score,score_hat, feat = model(img)
            loss = loss_fn(score, score_hat, feat, target)
        else:
            score,feat = model(img)
            loss = loss_fn(score, feat, target)

        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                               device=None):
    # Re-ID evaluator removed; verification is handled separately
    raise NotImplementedError("Re-ID evaluator removed in favor of verification eval.")


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query
):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    using_cal = cfg.MODEL.CAL
    verify_enabled = hasattr(cfg, 'VERIFY') and cfg.VERIFY.ENABLE
    pairs_file = cfg.VERIFY.PAIRS_FILE if verify_enabled else None
    verify_bs = cfg.VERIFY.BATCH_SIZE if verify_enabled else 128


    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fn, using_cal,device=device)
    timer = Timer(average=True)
    best_verify_acc = 0.0
    latest_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_latest.pth")
    best_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_best_acc.pth")
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] \nLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], 
                                engine.state.metrics['avg_acc'], scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    def _load_img(path, tfm):
        img = Image.open(path).convert('RGB')
        return tfm(img).unsqueeze(0)

    def _run_verification(current_model, pairs_txt, batch_size, device_str):
        if not pairs_txt or not os.path.isfile(pairs_txt):
            logging.getLogger("reid_baseline.train").warning(f"Pairs file not found: {pairs_txt}")
            return None
        module = current_model.module if hasattr(current_model, 'module') else current_model
        tfm = build_transforms(cfg, is_train=False)
        pairs = []
        with open(pairs_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b, y = line.split()
                pairs.append((a, b, int(y)))
        if not pairs:
            return None
        paths = sorted(set([p for a, b, _ in pairs for p in (a, b)]))
        feats = {}
        dev = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        current_model.eval()
        with torch.no_grad():
            batch_imgs, batch_paths = [], []
            for p in paths:
                try:
                    img = _load_img(p, tfm)
                except Exception:
                    continue
                batch_imgs.append(img)
                batch_paths.append(p)
                if len(batch_imgs) == batch_size:
                    x = torch.cat(batch_imgs, 0).to(dev)
                    f = module.extract_feat(x).cpu().numpy()
                    for bp, bf in zip(batch_paths, f):
                        feats[bp] = bf
                    batch_imgs, batch_paths = [], []
            if batch_imgs:
                x = torch.cat(batch_imgs, 0).to(dev)
                f = module.extract_feat(x).cpu().numpy()
                for bp, bf in zip(batch_paths, f):
                    feats[bp] = bf
        scores, labels = [], []
        for a, b, y in pairs:
            if a not in feats or b not in feats:
                continue
            fa, fb = feats[a], feats[b]
            scores.append(float(np.dot(fa, fb)))
            labels.append(y)
        if not scores:
            return None
        scores = np.array(scores)
        labels = np.array(labels)
        from sklearn import metrics as skm
        fpr, tpr, thr = skm.roc_curve(labels, scores)
        auc = skm.auc(fpr, tpr)
        best_thr_idx = np.argmax(tpr - fpr)
        thr_star = thr[best_thr_idx]
        preds = (scores >= thr_star).astype(int)
        acc = (preds == labels).mean()
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer_val = max(fpr[eer_idx], fnr[eer_idx])
        return {'acc': float(acc), 'auc': float(auc), 'eer': float(eer_val), 'thr': float(thr_star)}

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_latest(engine):
        try:
            torch.save(model, latest_path)
        except Exception as e:
            logger.warning(f"Failed to save latest model: {e}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def verify_and_maybe_save_best(engine):
        epoch = engine.state.epoch
        if not verify_enabled:
            return
        if (epoch % eval_period != 0) and (epoch != epochs):
            return
        res = _run_verification(model, pairs_file, verify_bs, device)
        if res is None:
            logger.warning("Verification skipped (no pairs or features).")
            return
        logger.info("Verification Results - Epoch {} | ACC: {:.4f}, AUC: {:.4f}, EER: {:.4f}, thr*: {:.4f}".format(
            epoch, res['acc'], res['auc'], res['eer'], res['thr']))
        print("Verification Results - Epoch {} | ACC: {:.4f}, AUC: {:.4f}, EER: {:.4f}, thr*: {:.4f}".format(
            epoch, res['acc'], res['auc'], res['eer'], res['thr']))
        nonlocal best_verify_acc
        if res['acc'] > best_verify_acc:
            best_verify_acc = res['acc']
            try:
                torch.save(model, best_path)
                logger.info("New best verification ACC {:.4f}. Saved to {}".format(best_verify_acc, best_path))
            except Exception as e:
                logger.warning(f"Failed to save best model: {e}")



    trainer.run(train_loader, max_epochs=epochs)
