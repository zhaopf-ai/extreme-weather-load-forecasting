import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MEKFOnlineAdapter:
    """Online MEKF adapter for parameter updates."""
    def __init__(
        self,
        model,
        R=1e-3,
        Q0=1e-5,
        mu_v=0.9,
        mu_p=0.9,
        lamb=0.99,
        delta=1e-6,
    ):
        self.model = model.to(device)
        self.model.eval()

        self.theta_vec, self.shapes = self._flatten_params()
        self.num_params = self.theta_vec.numel()

        self.P = torch.ones(self.num_params, device=device) * 1e-3
        self.Q = torch.ones(self.num_params, device=device) * Q0

        self.R = torch.tensor(float(R), device=device)
        self.mu_v = mu_v
        self.mu_p = mu_p
        self.lamb = lamb
        self.delta = delta

        self.V_prev = torch.zeros(self.num_params, device=device)

    def _flatten_params(self):
        vec, shapes = [], []
        for p in self.model.parameters():
            shapes.append(p.shape)
            vec.append(p.data.view(-1))
        theta = torch.cat(vec).detach().clone().to(device)
        return theta, shapes

    def _assign_params(self, theta_vec):
        idx = 0
        for p, shape in zip(self.model.parameters(), self.shapes):
            numel = p.numel()
            p.data.copy_(theta_vec[idx: idx + numel].view(shape))
            idx += numel

    def _compute_jacobian(self, pred):
        self.model.zero_grad()
        pred_scalar = pred.view(-1)[0]

        grads = torch.autograd.grad(
            pred_scalar,
            list(self.model.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        flat = []
        for p, g in zip(self.model.parameters(), grads):
            if g is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(g.contiguous().view(-1))

        return torch.cat(flat).detach()

    def step(self, weather, past, time, year, month, day, hour, target):
        self.model.eval()

        weather = weather.to(device)
        past = past.to(device)
        time = time.to(device)
        target = target.to(device)
        year, month, day, hour = (
            year.to(device),
            month.to(device),
            day.to(device),
            hour.to(device),
        )

        batch_size = weather.size(0)
        assert batch_size == 1

        ind = torch.arange(batch_size, device=device)
        lam = torch.ones(batch_size, 1, 1, 1, device=device)

        with torch.enable_grad():
            pred = self.model(
                weather,
                past,
                time,
                year,
                month,
                day,
                hour,
                ind,
                lam,
                Modal=0,
            )
            H_t = self._compute_jacobian(pred)

        y_hat_vec = pred.view(-1)
        y_t_vec = target.view(-1)

        y_hat = y_hat_vec[0]
        y_t = y_t_vec[0]
        e_t = y_t - y_hat


        P_prior = self.P + self.Q

        S_t = (H_t * P_prior * H_t).sum() + self.R
        K_t = P_prior * H_t / S_t

        V_star = K_t * e_t
        V_t = self.mu_v * self.V_prev + (1.0 - self.mu_v) * V_star
        self.V_prev = V_t.clone()

        self.theta_vec = (self.theta_vec + V_t).detach()
        self._assign_params(self.theta_vec)

        P_post = P_prior - K_t * H_t * P_prior
        P_raw = P_post / self.lamb + self.delta
        self.P = self.mu_p * self.P + (1.0 - self.mu_p) * P_raw

        return y_hat_vec.detach().cpu(), y_t_vec.detach().cpu()

