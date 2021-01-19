import torch.nn.functional as F

def distillation(y, teacher_scores, labels, T, alpha):
    kl_div = F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
    return kl_div * (T*T * 2. * alpha) + F.cross_entropy(y, labels, ignore_index=255) * (1. - alpha)