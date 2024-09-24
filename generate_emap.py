import os
import cv2
import math
import clip
import json
import numpy as np
from clip import tokenize
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize
import torch.nn.functional as F
from skimage.transform import resize as np_resize

import Game_MM_CLIP.clip as mm_clip
import CLIP_Surgery.clip as surgery_clip

from M2IB.scripts.clip_wrapper import ClipWrapper
from M2IB.scripts.methods import vision_heatmap_iba, text_heatmap_iba
from transformers import CLIPTokenizerFast

from tqdm import tqdm

_transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def imgprocess_keepsize(img, patch_size=[16, 16], scale_factor=1):
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)


# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, preprocess = clip.load("ViT-B/16", device=device)
mm_clipmodel, _ = mm_clip.load("ViT-B/16", device=device, jit=False)
surgery_model, _ = surgery_clip.load("CS-ViT-B/16", device=device)

# clipmodel, preprocess = clip.load("ViT-B/32", device=device)
# mm_clipmodel, _ = mm_clip.load("ViT-B/32", device=device, jit=False)
# surgery_model, _ = surgery_clip.load("CS-ViT-B/32", device=device)
# clipmodel.load_state_dict(torch.load("clip-imp-pretrained_128_6_after_4.pt", map_location=device))
# mm_clipmodel.load_state_dict(torch.load("clip-imp-pretrained_128_6_after_4.pt", map_location=device))
# surgery_model.load_state_dict(torch.load("clip-imp-pretrained_128_6_after_4.pt", map_location=device))

clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
m2ib_model = ClipWrapper(clipmodel)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # print("pred:", pred.shape) # [5,1]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    pred_top1 = pred[0,:]
    return acc, pred_top1


def generate_masks(input_size, N, s, p1):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))
    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = np_resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    masks = masks.reshape(-1, 1, *input_size)
    return torch.tensor(masks)


### RISE
def rise(model, image, txt_embedding, device, N=2000, s=8, p1=0.5):
    input_size = image.shape[-2:]
    masks = generate_masks(input_size, N, s, p1)
    batch_size = 50
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = image * masks
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc='Explaining'):
            image_features = model.encode_image(masked[i:min(i+batch_size, N)].to(device))
            image_features = F.normalize(image_features, dim=-1)
            preds.append((image_features @ txt_embedding.T).cpu())
            del image_features
    preds = torch.cat(preds, dim=0)
    sal = (preds * masks.reshape(N, -1)).sum(0).reshape(*input_size)
    sal = sal / N / p1
    return sal


### M2IB
def m2ib_clip_map(model, image, texts, device, vbeta=0.1, vvar=1, vlayer=9, tbeta=0.1, tvar=1, tlayer=9):
    text_ids = torch.tensor([clip_tokenizer.encode(texts, add_special_tokens=True)]).to(device)
    vmap = vision_heatmap_iba(text_ids, image, model, vlayer, vbeta, vvar)
    return vmap

def m2ib_clip_text_map(model, image, texts, device, vbeta=0.1, vvar=1, vlayer=9, tbeta=0.1, tvar=1, tlayer=9):
    text_ids = torch.tensor([clip_tokenizer.encode(texts, add_special_tokens=True)]).to(device)
    tmap = text_heatmap_iba(text_ids, image, model, tlayer, tbeta, tvar)
    return tmap


### CLIPSurgery
def clip_surgery_map(model, image, texts, device):
    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = surgery_clip.encode_text_with_prompt_ensemble(model, texts, device)

        # Apply feature surgery
        similarity = surgery_clip.clip_feature_surgery(image_features, text_features)
        similarity_map = surgery_clip.get_similarity_map(similarity[:, 1:, :], image.shape[-2:])
    return similarity_map


### GAME
def mm_interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1, flag="image", rollout=False):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    if flag == "image":
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if start_layer == -1: 
            # calculate index of last layer 
            start_layer = len(image_attn_blocks) - 1
        
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        attentions = []
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            avg_heads = (cam.sum(dim=0) / cam.shape[0]).detach()
            attentions.append(avg_heads.unsqueeze(0))
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]
        dim = int(image_relevance[0].numel() ** 0.5)
        image_relevance = image_relevance.reshape(batch_size, dim, dim)
        if rollout:
            return attentions
        return image_relevance
    
    if flag == "text":
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if start_layer_text == -1: 
            # calculate index of last layer 
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        attentions = []
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            avg_heads = (cam.sum(dim=0) / cam.shape[0]).detach()
            attentions.append(avg_heads.unsqueeze(0))
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text

        if rollout:
            return attentions
        return text_relevance

# compute rollout between attention layers
# copy codes from https://github.com/hila-chefer/Transformer-Explainability
def compute_rollout_attention(all_layer_matrices, start_layer=0, flag="image"):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    if flag=="text":
        return joint_attention
    if flag=="image":
        joint_attention = joint_attention[:, 0, 1:]
        dim = int(joint_attention[0].numel() ** 0.5)
        return joint_attention.reshape(batch_size, dim, dim)


def attention_layer(q, k, v, num_heads=1, attn_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling
    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn_output_weights += attn_mask
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_heads = torch.bmm(attn_output_weights, v)
    assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, attn_output_weights
    
def clip_encode_dense(x):
    vision_width = clipmodel.visual.transformer.width
    vision_heads = vision_width // 64
    clip_inres = clipmodel.visual.input_resolution
    clip_ksize = clipmodel.visual.conv1.kernel_size
    
    # modified from CLIP
    x = x.half()
    x = clipmodel.visual.conv1(x)  
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1) 
    x = x.permute(0, 2, 1) 
    class_embedding = clipmodel.visual.class_embedding.to(x.dtype)

    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

    pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = x + pos_embedding
    x = clipmodel.visual.ln_pre(x)
    
    x = x.permute(1, 0, 2)  # NLD -> LND
    x_in = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[:-1])(x)

    ##################
    # LastTR.attention
    targetTR = clipmodel.visual.transformer.resblocks[-1]
    x_before_attn = targetTR.ln_1(x_in)
    
    linear = torch._C._nn.linear    
    q, k, v = linear(x_before_attn, targetTR.attn.in_proj_weight, targetTR.attn.in_proj_bias).chunk(3, dim=-1)
    attn_output, attn = attention_layer(q, k, v, 1) #vision_heads
    x_after_attn = linear(attn_output, targetTR.attn.out_proj.weight, targetTR.attn.out_proj.bias)
    
    x = x_after_attn + x_in
    x_out = x + targetTR.mlp(targetTR.ln_2(x))

    x = x_out.permute(1, 0, 2)  # LND -> NLD
    x = clipmodel.visual.ln_post(x)
    x = x @ clipmodel.visual.proj
    
    ## ==== get lastv ==============
    with torch.no_grad():
        qkv = torch.stack((q, k, v), dim=0)
        qkv = linear(qkv, targetTR.attn.out_proj.weight, targetTR.attn.out_proj.bias)
        q_out, k_out, v_out = qkv[0], qkv[1], qkv[2]

        v_final = v_out + x_in
        v_final = v_final + targetTR.mlp(targetTR.ln_2(v_final))
        v_final = v_final.permute(1, 0, 2)
        v_final = clipmodel.visual.ln_post(v_final)
        v_final = v_final @ clipmodel.visual.proj
    ##############
    
    return x, v_final[:,1:], x_in, v, q_out, k_out, attn, attn_output, (feah, feaw)

    
def clip_encode_text_dense(text, n):
    x = clipmodel.token_embedding(text).type(clipmodel.dtype)  # [batch_size, n_ctx, d_model]
    attn_mask=clipmodel.build_attention_mask().to(dtype=x.dtype, device=x.device)
    x = x + clipmodel.positional_embedding.type(clipmodel.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x_in = torch.nn.Sequential(*clipmodel.transformer.resblocks[:-n])(x)

    #####################
    attns = []
    atten_outs = []
    vs = []
    q_outs = []
    k_outs = []
    for TR in clipmodel.transformer.resblocks[-n:]:
        x = TR.ln_1(x_in)
        linear = torch._C._nn.linear    
        q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
        attn_output, attn = attention_layer(q, k, v, 1, attn_mask=attn_mask) # transformer_heads=8
        attns.append(attn)
        atten_outs.append(attn_output)
        vs.append(v)
        x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)       
        x = x_after_attn + x_in
        x = x + TR.mlp(TR.ln_2(x))
        x_in = x
        
        with torch.no_grad():
            qkv = torch.stack((q, k, v), dim=0)
            qkv = linear(qkv, TR.attn.out_proj.weight, TR.attn.out_proj.bias)
            q_out, k_out, v_out = qkv[0], qkv[1], qkv[2] 
            q_outs.append(q_out)
            k_outs.append(k_out)
            
       
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clipmodel.ln_final(x).type(clipmodel.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clipmodel.text_projection
    
    return x, (q_outs, k_outs, vs), attns, atten_outs

def grad_eclip_text(c, q_outs, k_outs, lastvs, attn_outputs, eos_position, withksim=True):
    ## gradient on last attention output
    tmp_maps = []
    for q_out, k_out, lastv, attn_output in zip(q_outs, k_outs, lastvs, attn_outputs):
        grad = torch.autograd.grad(
            c,
            attn_output,
            retain_graph=True)[0]
        grad_cls = grad[eos_position,0,:]
        # just use the gradient on the cls token position  
        # cosine_qk = sim_qk(q_out, k_out)
        if withksim:
            q_cls = q_out[eos_position,0,:] 
            k_patch = k_out[:,0,:]
            q_cls = F.normalize(q_cls, dim=-1)
            k_patch = F.normalize(k_patch, dim=-1)
            cosine_qk = (q_cls * k_patch).sum(-1) 
            cosine_qk = (cosine_qk-cosine_qk.min()) / (cosine_qk.max()-cosine_qk.min())
            tmp_maps.append((grad_cls * lastv[:,0,:] * cosine_qk[:,None]).sum(-1))
        else:
            tmp_maps.append((grad_cls * lastv[:,0,:]).sum(-1))
        # print("[cosine_qk]:", cosine_qk.shape) # 77
        # tmp_maps.append((grad_cls * lastv[:,0,:] * cosine_qk[:,None]).sum(-1))  #  
    # grad_output*attn*v --> map
    emap_lastv = (F.relu_(torch.stack(tmp_maps, dim=0).sum(0))) # * cosine_qk[:,None]
    emap = emap_lastv[1:eos_position].flatten()
    emap = emap / emap.sum()
    
    return emap


def grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True):
    D = k_out.shape[-1]
    ## gradient on last attention output
    grad = torch.autograd.grad(
        c,
        att_output,
        retain_graph=True)[0]
    grad = grad.detach()
    grad_cls = grad[:1,0,:]
    if withksim:
        q_cls = q_out[:1,0,:]
        k_patch = k_out[1:,0,:]
        q_cls = F.normalize(q_cls, dim=-1)
        k_patch = F.normalize(k_patch, dim=-1)
        cosine_qk = (q_cls * k_patch).sum(-1) 
        cosine_qk = (cosine_qk-cosine_qk.min()) / (cosine_qk.max()-cosine_qk.min())
        emap_lastv = F.relu_((grad_cls * v[1:,0,:] * cosine_qk[:,None]).detach().sum(-1)) # 
    else:
        emap_lastv = F.relu_((grad_cls * v[1:,0,:]).detach().sum(-1)) 
    return emap_lastv.reshape(*map_size)
    
### Grad-CAM
def grad_cam(c, layer_feat, map_size):
    ## GRAD-CAM: use the feature outputs of the last second attention layer
    grad = torch.autograd.grad(
        c,
        layer_feat,
        retain_graph=True)[0]
    grad = grad.detach()
    grad_weight = grad.mean(0, keepdim=True)
    grad_cam = F.relu_((grad_weight * layer_feat[1:,0,:]).detach().sum(-1)) 
    return grad_cam.reshape(*map_size)

### MaskCLIP
def mask_clip(txt_feats, v_final, k_out, map_size):
    ## similarity between text prompt and v_out
    v_final = F.normalize(v_final, dim=-1)
    cosine_v = (v_final @ txt_feats)[0].transpose(1,0)
    # print("[position similarity (cosine v)]:", cosine_v.shape)
    # print("[map size]:", map_size)
    k_cls = k_out[:1,0,:]
    k_patch = k_out[1:,0,:]
    k_cls = F.normalize(k_cls, dim=-1)
    k_patch = F.normalize(k_patch, dim=-1)
    cosine_qk = (k_cls * k_patch).sum(-1)

    sim_v = cosine_v * cosine_qk[None,:]
    return sim_v.detach().reshape(-1, *map_size)

def save_map(image, emap, resize, path, tag):
    emap -= emap.min()
    emap /= emap.max()
    emap = resize(emap.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((emap*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    np.save(os.path.join(path, "npy_hm", "{}.npy".format(tag)), emap)
    cv2.imwrite(os.path.join(path, "color_visual", "{}.png".format(tag)), c_ret[:,:,::-1])


if __name__ == '__main__':
    from clip_utils import build_zero_shot_classifier
    from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    data_path = "../imagenet-1k/val/"
    save_path = "../grad_eclip/imagenet-1k/"

    zero_shot_weights = build_zero_shot_classifier(
        clipmodel,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True
        )
    print("[class name embeddings]:", zero_shot_weights.shape)  # [512, 1000]

    folders = os.listdir(data_path)

    top1, top5, top10, n = 0., 0., 0., 0.

    with open("imagenet_class_index.json", "r") as ff:
        class_dict = json.load(ff)
        for label, values in list(class_dict.items()):
            label = int(label)
            folder = values[0]
            
            files = os.listdir(data_path+folder)
            if not os.path.exists(save_path+"npy_hm/"+folder):
                os.makedirs(save_path+"npy_hm/"+folder)
            if not os.path.exists(save_path+"color_visual/"+folder):
                os.makedirs(save_path+"color_visual/"+folder)

            print("Start: Processing the {}th folder, target class name: {}".format(label, IMAGENET_CLASSNAMES[label]))
            for f in files:
                img_name = f.split(".")[0]
                img = Image.open(os.path.join(data_path, folder, f)).convert("RGB")
                w, h = img.size
                # in case there is too large image
                if min(w,h) > 640:
                    scale = min(w,h) / 640
                    hs = int(h/scale)
                    ws = int(w/scale)
                    # print(img_name, w, h, ws, hs)
                    img = img.resize((ws,hs))
                w, h = img.size
                resize = T.Resize((h,w))
                # make prediction
                with torch.no_grad():
                    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
                    img_clip_embedding = clipmodel.encode_image(img_clipreprocess)
                    logits = 100. * img_clip_embedding @ zero_shot_weights
                    target = torch.tensor([label]).to(device)
                    [acc1, acc5, acc10], pred_top1 = accuracy(logits, target, topk=(1, 5, 10))
                    top1 += acc1
                    top5 += acc5
                    top10 += acc10
                    n += img_clipreprocess.size(0)            
                
                ########## explanation maps
                class_tags = [label, pred_top1.item()]
                target_pred = zero_shot_weights[:, torch.cat([target, pred_top1])]  # [512, 2]
                img_keepsized = imgprocess_keepsize(img).to(device).unsqueeze(0)
                outputs, v_final, last_input, v, q_out, k_out,\
                    attn, att_output, map_size = clip_encode_dense(img_keepsized)

                # self_attn
                # self_attn = attn[0,:1,1:].detach().reshape(*map_size)
                # maskclip            
                # maskclips = mask_clip(target_pred, v_final, k_out, map_size)

                # cosine
                img_embedding = F.normalize(outputs[:,0], dim=-1)
                cosines = (img_embedding @ target_pred)[0]   # [2]
                tags = ["gt", "pred"]
                image = np.asarray(img.copy())
                for i, c in enumerate(cosines):
                    # grad-eclip
                    eclip = grad_eclip(c, k_out, v, att_output, map_size)
                    save_map(image, eclip, resize, save_path, "{}/{}_eclip-wo-ksim_{}_{}".format(folder, img_name, tags[i], class_tags[i]))
                    
                    # game-interprete
                    # text_tokenized = mm_clip.tokenize(IMAGENET_CLASSNAMES[class_tags[i]]).to(device)
                    # R_image = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device)
                    # save_map(image, R_image, resize, save_path, "{}/{}_game_{}_{}".format(folder, img_name, tags[i], class_tags[i]))

                    # rollout
                    # text_tokenized = mm_clip.tokenize(IMAGENET_CLASSNAMES[class_tags[i]]).to(device)
                    # attentions = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device, rollout=True)
                    # rollout_image = compute_rollout_attention(attentions)[0]
                    # save_map(image, rollout_image, resize, save_path, "{}/{}_rollout_{}_{}".format(folder, img_name, tags[i], class_tags[i]))                    
                    
                    # grad-cam
                    # gradcam = grad_cam(c, last_input, map_size)
                    # save_map(image, gradcam, resize, save_path, "{}/{}_gradcam_{}_{}".format(folder, img_name, tags[i], class_tags[i]))
                    # save_map(image, maskclips[i], resize, save_path, "{}/{}_maskclip_{}_{}".format(folder, img_name, tags[i], class_tags[i]))
                # save_map(image, self_attn, resize, save_path, "{}/{}_selfattn".format(folder, img_name))
                
        top1 = (top1 / n)
        top5 = (top5 / n)
        top10 = (top10 / n)
        print("[clip accuracy] Top1:{:.4f}, Top5:{:.4f}, Top10:{:.4f}".format(top1, top5, top10))





    