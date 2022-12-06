import torch
import torch.nn.functional as F


def dandr_loss(logits_student, logits_teacher, target, alpha, beta, temperature, detach_target=True):
    if detach_target:
        logits_teacher = logits_teacher.detach()
    index_fg = (target != (logits_teacher.shape[1] - 1))
    index_bg = (target == (logits_teacher.shape[1] - 1))

    gt_mask = _get_target_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_teacher = F.softmax(logits_teacher[index_fg] / temperature, dim=1)

    p_non_target_pos_teacher = (pred_teacher * other_mask[index_fg]).sum(1, keepdims=True)[:, 0]

    non_target_logits_teacher = logits_teacher - 1000.0 * gt_mask #.type(torch.float64)
    non_target_logits_student = logits_student - 1000.0 * gt_mask #.type(torch.float64)

    bg_mask = _get_bg_mask(non_target_logits_teacher)
    non_bg_mask = _get_non_bg_mask(non_target_logits_teacher)
    non_target_pred_student = F.softmax(non_target_logits_student / temperature, dim=1)
    non_target_pred_teacher = F.softmax(non_target_logits_teacher / temperature, dim=1)

    p_fbd_student = cat_mask(non_target_pred_student, bg_mask, non_bg_mask)
    p_fbd_teacher = cat_mask(non_target_pred_teacher, bg_mask, non_bg_mask)

    log_p_fbd_student = torch.log(p_fbd_student)
    loss_fbd =(
            F.kl_div(log_p_fbd_student, p_fbd_teacher, reduction='none').sum(1)
            * (temperature ** 2)
    )

    p_fcd_teacher = F.softmax(
        non_target_logits_teacher / temperature - 1000 * bg_mask, dim=1
    )
    log_p_fcd_student = F.log_softmax(
        non_target_logits_student / temperature - 1000 * bg_mask, dim=1
    )
    loss_fcd = p_fbd_teacher[:, 1] * (
        F.kl_div(log_p_fcd_student, p_fcd_teacher, reduction='none').sum(1)
        * (temperature**2)
    )

    loss = alpha * torch.mean(p_non_target_pos_teacher * loss_fbd[index_fg]) \
            + beta * torch.mean(loss_fbd[index_bg]) \
            + torch.mean(p_non_target_pos_teacher * loss_fcd[index_fg])\
            + torch.mean(loss_fcd[index_bg])

    return loss


def _get_target_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    mask[:, -1] = 0
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    mask[:, -1] = 1
    return mask

def _get_bg_mask(logits):
    mask = torch.zeros_like(logits)
    mask[:, -1] = 1
    mask = mask.bool()
    return mask

def _get_non_bg_mask(logits):
    mask = torch.ones_like(logits)
    mask[:, -1] = 0
    mask = mask.bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


