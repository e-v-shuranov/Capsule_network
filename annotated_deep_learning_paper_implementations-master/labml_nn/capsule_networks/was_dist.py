
import math
import torch
import torch.linalg as linalg
import torch.nn.functional as F

def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()
def Wasserstein_simple(p,q):


    # Sort in descending order to align probabilities
    student = q.sort(dim=-1, descending=True).values
    teacher = p.sort(dim=-1, descending=True).values

    size = min(student.size(2), teacher.size(2))

    # Pad to get same vocabulary size
    # diff_size = student.size(2) - teacher.size(2)
    # if diff_size > 0:
    #     teacher = F.pad(teacher, (0, diff_size), value=0)
    # elif diff_size < 0:
    #     student = F.pad(student, (0, abs(diff_size)), value=0)

    # distillation_loss = calculate_2_wasserstein_dist(teacher, student)
    # return distillation_loss

    distillation_loss1 = abs(student[:, :, :size] - teacher[:, :, :size]).mean(-1)
    size = min(q.size(2), p.size(2))
    distillation_loss = abs(q[:, :, :size] - p[:, :, :size]).mean(-1)

    # # distillation_loss = torch.zeros(student.size(1), device=student.device)
    # distillation_loss = torch.zeros(student.size(1), device=student.device)
    #
    # for i in range(student.size(1)):
    #     # size = min(student.size(2), teacher.size(2))  # it looks unnesesary it is equal after padding
    #     distillation_loss[i] = abs(student[0, i, :student.size(2)] - teacher[0, i, :student.size(2)]).mean(-1)
    #     # distillation_loss[i] = abs(student[0,i,:size] - teacher[0,i,:size]).sum(-1).mean(-1)
    # # distillation_loss = distillation_loss.mean()
    # distillation_loss = distillation_loss.unsqueeze(0)

    return distillation_loss

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def uld_simplify_loss(p, p_on_student_tokenizer, logq, q, tokens, mask, alpha=1, betta=1.0):
    """
        ULDloss = CE[p,q] +  α*W_loss(p,q)
               where CE[p, q] = -log(q) is cross-entropy loss,
               p: teacher probs (over teacher vocab)
               q: student probs (over stud vocab)
               alpha: tunable ULD  weight α

        logq: student log-probs (over stud vocab)  to calculate CE
        tokens: generated tokens
        mask: specifies loss action-masking
    """
    # ULD part
    W_loss = Wasserstein_simple(p, q)
    #     W_loss = wasserstein_distance(p.cpu().detach().numpy(), q.detach().cpu().numpy())   # not enough memory
    print(f' uld_simplify_loss W_loss', W_loss, 'init')

    # SLIM part
    kd_loss = -torch.sum(p_on_student_tokenizer * logq, axis=-1)
    token_logq = logq.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    token_p = p_on_student_tokenizer.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    ce_loss = -token_logq
    kd_weight = alpha * torch.where(
        token_p < 1,
        torch.where(
            token_p > 0,
            -torch.expm1(-token_logq.detach() / token_p.log()),
            0.),
        1.)
    print("ce_loss", ce_loss, "kd_weight", kd_weight, "kd_loss", kd_loss, "W_loss", W_loss)
    loss = ce_loss + kd_weight * kd_loss + betta * W_loss
    #     loss = ce_loss + kd_weight * kd_loss
    print("loss", loss)
    print("mask", mask)
    # average
    ce_loss = masked_mean(ce_loss, mask)
    kd_loss = masked_mean(kd_loss, mask)
    kd_weight = masked_mean(kd_weight, mask)
    W_loss = masked_mean(W_loss, mask)
    loss = masked_mean(loss, mask)
    print("uld_simplify_loss")

    print("loss", loss)
    print("ce_loss", ce_loss)
    print("kd_loss", kd_loss)
    print("kd_weight", kd_weight)
    print("W_loss", W_loss)


    return loss, ce_loss, kd_loss, kd_weight, W_loss




def main():
    from scipy.stats import wasserstein_distance  # ULD
    import numpy as np
    logits_p = torch.rand(1, 4096, 100888)
    p = F.softmax(logits_p[:, :-1, :], dim=-1)
    logits = torch.rand(1, 4096, 48000)
    q = F.softmax(logits[:, :-1, :], dim=-1)
    logq = F.log_softmax(logits[:, :-1, :], dim=-1)

    p_on_student_tokenizer = torch.rand(1, 4095, 48000)
    # logq =torch.rand(1, 4095, 48000)
    tokens = torch.randint(high  = 48000, size  = (1, 4096)) #, dtype = torch.dtype("int64"))

    mask = torch.rand(1, 4096)
    # qq = np.random.rand(1, 4095, 48000, 4095, 48000)
    # print("qq.shape:", qq.shape)
    # if p.shape() != q.shape():
    print("p.shape and q.shape", p.shape , q.shape)

    loss, ce_loss, kd_loss, kd_weight, ULD_loss,  = uld_simplify_loss(p, p_on_student_tokenizer,  logq, q, tokens[:, 1:], mask[:, 1:])
    print("--- main---")

    # W_loss = Wasserstein_simple(p,q)
    # W_loss = wasserstein_distance(p.cpu().detach().numpy(), q.detach().cpu().numpy())
    print("loss", loss)
    print("ce_loss", ce_loss)
    print("kd_loss", kd_loss)
    print("kd_weight", kd_weight)
    print("ULD_loss", ULD_loss)

    print(loss.shape)
    print(loss)


if __name__ == '__main__':
    main()