import torch.nn.functional as F

def distillation(y, teacher_scores, labels, T, alpha):
    kl_div = F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
    return kl_div * (T*T * 2. * alpha) + F.cross_entropy(y, labels, ignore_index=255) * (1. - alpha)

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def match_at_layers(teacher_layers, student_layers):
    j = 0
    n = len(student_layers)
    m = len(teacher_layers)
    
    at_student = []
    at_teacher = []
    
    for i in range(n):
        size_student = student_layers[i].shape[2]
        if i < n-1 and size_student == student_layers[i+1].shape[2]:
            continue
        for j in range(m):
            size_teacher = teacher_layers[j].shape[2]
            if j < m-1 and size_teacher == teacher_layers[j+1].shape[2] and size_teacher != size_student:
                continue
            
            if (size_student == size_teacher):
                at_student.append(student_layers[i])
                at_teacher.append(teacher_layers[j])
                break
            
    return at_teacher, at_student
    