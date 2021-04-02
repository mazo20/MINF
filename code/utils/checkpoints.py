import torch

def save_ckpt(path, opts, model, optimizer, scheduler, best_score, epoch):
        """ save current model
        """
        
        # root = os.path.join(path, 'output')
        root = os.path.join(path, 'output', opts.results_root)
        if not os.path.exists(root):
            os.mkdir(root)
        
        path = root + '/%s_%s_os%d_%d.pth' % (opts.model, opts.dataset, opts.output_stride, opts.index)
        
        torch.save({
            "epoch":           epoch,
            "best_score":      best_score,
            "model_state":     model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)