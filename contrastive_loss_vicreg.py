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
        self.lamda = 25
        self.mu = 100
        self.nu = 1

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

            # variance

            std_z_view1 = torch.sqrt(z_view1_vec.var(dim=0) + 1e-04)
            std_z_view2 = torch.sqrt(z_view2_vec.var(dim=0) + 1e-04)
            var_loss = torch.mean(torch.relu(1 - std_z_view1)) + torch.mean(
                torch.relu(1 - std_z_view2)
            )
            # invariance
            dist_loss = (torch.norm(z_view1_vec - z_view2_vec, dim=0) ** 2).mean()

            # Covariance matrix
            z_view1_vec = z_view1_vec.squeeze(2) - z_view1_vec.mean(dim=1)
            z_view2_vec = z_view2_vec.squeeze(2) - z_view2_vec.mean(dim=1)
            cov_z_a = (z_view1_vec.T @ z_view1_vec) / (z_view1_vec.shape[1] - 1)
            cov_z_b = (z_view2_vec.T @ z_view2_vec) / (z_view2_vec.shape[1] - 1)
            cov_z_a[range(cov_z_a.shape[0]), range(cov_z_a.shape[0])] = 0
            cov_z_b[range(cov_z_b.shape[0]), range(cov_z_b.shape[0])] = 0

            cov_loss = (
                cov_z_a.pow(2).sum() / z_view1_vec.shape[0]
                + cov_z_b.pow(2).sum() / z_view2_vec.shape[0]
            )

            loss += self.lamda * dist_loss + self.mu * var_loss + self.nu * cov_loss

        return (
            loss / (i + 1),
            pos_corr / (i + 1),
            neg_corr / (i + 1),
            dist_loss / (i + 1),
            var_loss / (i + 1),
            cov_loss / (i + 1),
        )
