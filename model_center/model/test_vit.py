import torch 
import bmtrain as bmt
import torch.nn.functional as F
from model_center.model import VisionTransformer
from model_center.model import VitConfig
#from vit_base import VisionTransformer2
# from model_center.layer import PatchEmbedding
# from vit_base import PatchEmbed
# from vit_base import Conv2d
# from vit_base import interpolate_pos_embed
bmt.init_distributed(
    seed = 0,
)
torch.manual_seed(1234)
num_layers = 12
Config = VitConfig.from_json_file("/root/data/openbmb/ModelCenter-main/configs/vit/config.json")
model = VisionTransformer(Config)
# model2 = VisionTransformer2(img_size=256, patch_size=16, 
#                             embed_dim=768, depth=12, num_heads=12, 
#                             mlp_ratio=4, qkv_bias=True,  
#                             dtype=torch.float)
# state_dict=torch.load("./pytorch_model.pt")['model']
# for i in state_dict:
#     state_dict[i]=state_dict[i].to(torch.device("cuda:0"))
# pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], model2)
# state_dict['pos_embed'] = pos_embed_reshaped

bmt.load(model,"/root/data/openbmb/ModelCenter-main/model_center/model/pytorch_model.pt",strict=False)
from IPython import embed ; embed()
# def lookup_output(model):
#     for name, layer in model.named_children():
#         layer.__name__ = name
#         if name in ["blocks",'0','layers','self_att','attn','self_attention']:
#             lookup_output(layer)
#         layer.register_forward_hook(
#             lambda layer, _, output: print(f"{layer.__name__}: {output.max()}")
#         )
# lookup_output(model)
# lookup_output(model2)


reanme_dic = {'cls_token': 'cls_token',
 'pos_embed': 'pos_embed',
 'patch_embed.proj.weight': 'patch_embed.proj.weight',
 'patch_embed.proj.bias': 'patch_embed.proj.bias',
 'blocks.output_layernorm.weight': 'norm.weight',
 'blocks.output_layernorm.bias': 'norm.bias',
 'head.weight': 'head.weight',
 'head.bias': 'head.bias'}
state_dic1 = {}
state_dic2 = model2.state_dict()
for idx in range(num_layers):
    reanme_dic[f'blocks.layers.{idx}.self_att.layernorm_before_attention.weight'] = f'blocks.{idx}.norm1.weight'
    reanme_dic[f'blocks.layers.{idx}.self_att.layernorm_before_attention.bias'] = f'blocks.{idx}.norm1.bias'
    reanme_dic[f'blocks.layers.{idx}.self_att.self_attention.attention_out.weight'] = f'blocks.{idx}.attn.proj.weight'
    reanme_dic[f'blocks.layers.{idx}.self_att.self_attention.attention_out.bias'] = f'blocks.{idx}.attn.proj.bias'
    reanme_dic[f'blocks.layers.{idx}.ffn.layernorm_before_ffn.weight'] = f'blocks.{idx}.norm2.weight'
    reanme_dic[f'blocks.layers.{idx}.ffn.layernorm_before_ffn.bias'] = f'blocks.{idx}.norm2.bias'
    reanme_dic[f'blocks.layers.{idx}.ffn.ffn.w_in.w.weight'] = f'blocks.{idx}.mlp.fc1.weight'
    reanme_dic[f'blocks.layers.{idx}.ffn.ffn.w_in.w.bias'] = f'blocks.{idx}.mlp.fc1.bias'
    reanme_dic[f'blocks.layers.{idx}.ffn.ffn.w_out.weight'] = f'blocks.{idx}.mlp.fc2.weight'
    reanme_dic[f'blocks.layers.{idx}.ffn.ffn.w_out.bias'] = f'blocks.{idx}.mlp.fc2.bias'
    qkv_weight=state_dic2[f'blocks.{idx}.attn.qkv.weight'].chunk(3,dim=0)
    qkv_bias=state_dic2[f'blocks.{idx}.attn.qkv.bias'].chunk(3,dim=0)
    for i in range(3):
        state_dic1[f'blocks.layers.{idx}.self_att.self_attention.project_{"qkv"[i]}.weight']=qkv_weight[i]
        state_dic1[f'blocks.layers.{idx}.self_att.self_attention.project_{"qkv"[i]}.bias']=qkv_bias[i]
    # state_dic2[f'blocks.{idx}.attn.qkv.weight'] = torch.cat([state_dic1[f'blocks.layers.{idx}.self_att.self_attention.project_{i}.weight'] for i in "qkv" ],0)
    # state_dic2[f'blocks.{idx}.attn.qkv.bias'] = torch.cat([state_dic1[f'blocks.layers.{idx}.self_att.self_attention.project_{i}.bias'] for i in "qkv" ],0)
# reanme_dic=dict(reanme_dic.values,reanme_dic.keys())
for i in reanme_dic:
    state_dic1[i]=state_dic2[reanme_dic[i]]
for i in state_dic1:
    state_dic1[i] = state_dic1[i].cuda()
# model.load_state_dict(state_dic1)
i  =  torch.randn((16,3,256,256),dtype = torch.float).cuda()
from IPython import embed;embed()


# model predicts one of the 1000 ImageNet classes
