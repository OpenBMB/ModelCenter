import torch 
import bmtrain as bmt
import torch.nn.functional as F
from model_center.model import VisionTransformer
from model_center.model import VitConfig
# c = torch.load('/data/home/scy0377/cqy/e.pth', map_location='cpu').cuda()
c = torch.randn((32,3,512,512)).cuda().half()
# model = VisionTransformer.from_pretrained("vit-base_patch16_224")

loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
target = torch.zeros(32).long().cuda()
import time
t_s = time.time()
# with torch.no_grad():
print('begin')
for x in range(128):
    optimizer.zero_grad()
    e=model(c)
    loss = loss_func(e, target)
    loss.backward()
    optimizer.step()
    
t_e=time.time()
print(t_e-t_s)
# # # print(e.shape)
# # # print(e[1][1][:10])