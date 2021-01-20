import torch

def save_ckpt(path, opts, model, optimizer, scheduler, best_score, epoch):
        """ save current model
        """
        
        path = path + '_%s_%s_os%d_%d.pth' % (opts.model, opts.dataset, opts.output_stride, opts.random_seed)
        
        torch.save({
            "epoch": epoch,
            "best_score": best_score,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            
        }, path)
        print("Model saved as %s" % path)