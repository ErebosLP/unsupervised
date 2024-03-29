import torch
import numpy as np
import ipdb


class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, factor_pos, factor_dist=0.8, examples=200 * 200):
        torch.nn.Module.__init__(self)
        self.factor_pos = factor_pos
        self.loss = torch.nn.BCELoss()
        self.factor_dist = factor_dist
        self.examples = examples

    def forward(self, views_1, views_2, img):
        torch.cuda.empty_cache()
        img = img.to("cuda")
        loss = 0
        pos_corr = 0
        neg_corr = 0
        # sim_all = torch.zeros((self.neg_examples + 1)).cuda()
        batch, c, h, w = views_1[0].unsqueeze(0).size()
        height = np.floor(np.arange(h * w) / w).astype(int)
        width = np.floor(np.arange(h * w) % w).astype(int)
        max_euc_dist = torch.norm(torch.tensor([256, 256], dtype=float).to("cuda"))
        max_rgb_dist = torch.sqrt(torch.tensor([3.0]).to("cuda"))
        view_1_norm = torch.nn.functional.normalize(views_1, p=2, dim=1)
        view_2_norm = torch.nn.functional.normalize(views_2, p=2, dim=1)
        for i in range(views_1.shape[0]):
            z_view1 = view_1_norm[i].unsqueeze(0)
            z_view2 = view_2_norm[i].unsqueeze(0)
            ########################################################################################################################################################################
            ########################################################################################################################################################################
            z_view1_vec = (
                torch.reshape(z_view1, [batch, c, h * w]).squeeze(0).unsqueeze(2)
            )
            z_view2_vec = (
                torch.reshape(z_view2, [batch, c, h * w]).squeeze(0).unsqueeze(2)
            )

            idx = np.zeros(z_view1_vec.shape[1], bool)
            idx[: self.examples] = 1
            idx = np.random.permutation(idx)

            z_view1_vec = z_view1_vec[:, idx, :]
            z_view2_vec = z_view2_vec[:, idx, :]
            mat = z_view1_vec.squeeze(2).T @ z_view2_vec.squeeze(2)

            mat[range(mat.shape[0]), range(mat.shape[0])] -= 1
            mat[range(mat.shape[0]), range(mat.shape[0])] *= self.factor_pos
            loss += (mat).pow(2).sum() / (mat.shape[0] ** 2) * 100

            pos_corr += (torch.diagonal(mat).sum() / self.factor_pos / mat.shape[0]) + 1
            neg_corr += (mat.sum() - (pos_corr * mat.shape[0])) / (
                mat.shape[0] * (mat.shape[0] - 1)
            )
        return loss / (i + 1), pos_corr / (i + 1), neg_corr / (i + 1), 0, 0, 0
