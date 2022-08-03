import torch
batch_size = 2
L_x = 2
L_y = 4
hidden_size = 10
txt_emb = torch.randn([batch_size,L_x,hidden_size])
txt_mask = torch.zeros([batch_size,L_x]).bool()
image_emb = torch.randn([batch_size,L_y,hidden_size])
image_mask = torch.ones([batch_size,L_y]).bool()
joint = txt_mask.unsqueeze(-1) | image_mask.unsqueeze(-2)
print(joint.shape)
print(joint)

x_len = (~txt_mask).int().sum(dim=1)
y_len = (~image_mask).int().sum(dim=1)
sigma = torch.ones([batch_size, L_x]) / x_len.unsqueeze(1)
print(x_len)
print(sigma.shape)
print(sigma)