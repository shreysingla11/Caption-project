import torch 
import torch.nn as nn 

class GLOCaptions(nn.Module):
    def __init__(self, glo, decoder, device='cuda', glo_dim=512, embed_method='fixed_steps', embed_step_size=1e-3, embed_num_steps=500, embed_loss_thresh=0.5, **kwargs):
        super().__init__(**kwargs)
        self.glo = glo
        self.glo_dim = glo_dim
        self.decoder = decoder
        self.device = device 
        self.embed_func, self.embed_args = self.get_embed_method(embed_method, embed_step_size, embed_num_steps, embed_loss_thresh)

    def get_embed_method(self, embed_method, step_size, num_steps, loss_thresh):
        kwargs = {'step_size': step_size}
        method = None 
        if embed_method == 'fixed_steps':
            kwargs['num_steps'] = num_steps
            method = self.fixed_steps_glo 
        elif embed_method == 'loss':
            kwargs['loss_thresh'] = loss_thresh
            method = self.loss_glo
        else:
            raise ValueError(f'Invalid embedding method "{embed_method}"')

    # TODO: ashutosh@sathe-pc/01-05-2022:15:33:41
    # Unify the 2 maybe ? Or is it better to have them separate ?
    def fixed_steps_glo(self, images, num_steps, step_size):
        def project_l2_ball(z):
            # TODO: ashutosh@sathe-pc/01-05-2022:15:34:49
            # Not sure if `z / torch.maximum(torch.sqrt(torch.sum(z**2, dim=1))[:, None], 1)` looks better style wise
            return z / torch.maximum(torch.sqrt(torch.sum(z**2, dim=1)).unsqueeze(-1), 1)
        n = images.size(0)
        self.glo.eval()
        z = torch.randn(n, self.glo_dim, requires_grad=True).to(device)
        z = project_l2_ball(z)
        mse = nn.MSELoss()
        optim = torch.optim.SGD(z, lr=step_size)
        for _ in range(num_steps):
            optim.zero_grad()
            recon = self.glo(z)
            loss = mse(recon, images)
            loss.backward()
            optim.step()
            z = project_l2_ball(z)
        return z

    def loss_glo(self, images, loss_thresh, step_size):
        def project_l2_ball(z):
            # TODO: ashutosh@sathe-pc/01-05-2022:15:34:49
            # Not sure if `z / torch.maximum(torch.sqrt(torch.sum(z**2, dim=1))[:, None], 1)` looks better style wise
            return z / torch.maximum(torch.sqrt(torch.sum(z**2, dim=1)).unsqueeze(-1), 1)
        n = images.size(0)
        self.glo.eval()
        z = torch.randn(n, self.glo_dim, requires_grad=True).to(device)
        z = project_l2_ball(z)
        mse = nn.MSELoss()
        optim = torch.optim.SGD(z, lr=step_size)
        next_step = True 
        while next_step:
            optim.zero_grad()
            recon = self.glo(z)
            loss = mse(recon, images)
            loss.backward()
            optim.step()
            z = project_l2_ball(z)
            next_step = loss.item() > loss_thresh
        return z
    
    def forward(self, batch):
        images = batch.pop('images')
        z = self.embed_func(images, **self.embed_args)
        return self.decoder(inputs_embeds = z.unsqueeze(dim=1), **batch)
