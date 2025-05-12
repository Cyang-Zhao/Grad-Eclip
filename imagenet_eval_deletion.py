import os
import json
import glob
import torch
import clip
from tqdm import tqdm
from clip import tokenize
from clip_utils import build_zero_shot_classifier
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

from PIL import Image
import numpy as np
from torchvision.transforms import Resize
import torch.nn.functional as F

from generate_emap import clipmodel, preprocess, imgprocess_keepsize, mm_clipmodel, mm_interpret, \
    clip_encode_dense, grad_eclip, grad_cam, mask_clip, compute_rollout_attention, surgery_model, clip_surgery_map, \
    m2ib_model, m2ib_clip_map

import Game_MM_CLIP.clip as mm_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # print("pred:", pred.shape) # [5, 10] 
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = [correct[:k,:].float().sum(0).cpu() for k in topk]
    pred_top1 = pred[0,:]
    return acc, pred_top1

def make_grids(h, w):
    shifts_x = torch.arange(
        0, w, 1)
    shifts_y = torch.arange(
        0, h, 1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids = torch.stack((shift_x, shift_y), dim=1)
    return grids

def random_pixel(image, poses):
    # adjust the size of perturbation each step based on the size of object size 
    h,w, _ = image.shape 
    random_patch = torch.rand(len(poses), 3) * 255.
    xs, ys = zip(*poses)
    image[ys, xs, :] = random_patch
    return image

def delection_process(image, heatmap, L, cal_gap, del_path, img_name):
    image_array = np.array(image).copy()
    image_array.setflags(write=1)
    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    area = h*w
    pixel_once = max(1, int(area/(2*L)))

    del_imgs = []
    for step in range(1,L+1):
        image_array = random_pixel(image_array, grids[order[(step-1)*pixel_once:step*pixel_once]])
        if step%cal_gap == 0:
            pil_image = Image.fromarray(np.uint8(image_array))
            img_clipreprocess = preprocess(pil_image).to(device).unsqueeze(0)
            del_imgs.append(img_clipreprocess)
            # pil_image.save(del_path+'/{}_{}.jpg'.format(img_name, step))
    return torch.cat(del_imgs, dim=0)


def generate_hm(hm_type, img, gt, pred, resize):
    img_keepsized = imgprocess_keepsize(img).to(device).unsqueeze(0)
    outputs, v_final, last_input, v, q_out, k_out,\
        attn, att_output, map_size = clip_encode_dense(img_keepsized)
    img_embedding = F.normalize(outputs[:,0], dim=-1)
    if "gt" in hm_type:
        exp_target = gt
        txt_embedding = zero_shot_weights[:, gt]
        cosine = (img_embedding @ txt_embedding)[0]
    elif "pred" in hm_type:
        exp_target = pred
        txt_embedding = zero_shot_weights[:, pred]
        cosine = (img_embedding @ txt_embedding)[0]
    else:
        None

    if hm_type == "selfattn":
        emap = attn[0,:1,1:].detach().reshape(*map_size)
    elif "gradcam" in hm_type:
        emap = grad_cam(cosine, last_input, map_size)
    elif "maskclip" in hm_type:
        emap = mask_clip(txt_embedding.unsqueeze(-1), v_final, k_out, map_size)[0]
    elif "eclip" in hm_type:
        emap = grad_eclip(cosine, q_out, k_out, v, att_output, map_size, withksim=False) \
            if "wo-ksim" in hm_type else grad_eclip(cosine, q_out, k_out, v, att_output, map_size, withksim=True)
    elif "game" in hm_type:
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(IMAGENET_CLASSNAMES[exp_target]).to(device)
        emap = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device)[0]
    elif "rollout" in hm_type:
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(IMAGENET_CLASSNAMES[exp_target]).to(device)   
        attentions = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device, rollout=True)      
        emap = compute_rollout_attention(attentions)[0]
    elif "surgery" in hm_type:
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
        all_texts.insert(0, IMAGENET_CLASSNAMES[exp_target])
        emap = clip_surgery_map(model=surgery_model, image=img_clipreprocess, texts=all_texts, device=device)[0,:,:,0]
    elif "m2ib" in hm_type:
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        emap = m2ib_clip_map(model=m2ib_model, image=img_clipreprocess, texts=IMAGENET_CLASSNAMES[exp_target], device=device)
        emap = torch.tensor(emap)


    emap -= emap.min()
    emap /= emap.max()
    emap = resize(emap.unsqueeze(0))[0].cpu().numpy()
    return emap

if __name__ == '__main__':
    data_path = "./data/imagenet/val/"
    del_path = './data/imagenet/del_samples/' ### for debug, saving deleting samples

    # hm_types = ['eclip-wo-ksim_gt', 'eclip-wo-ksim_pred', 'eclip_gt', 'eclip_pred', 'game_gt', 'game_pred',
    #         'gradcam_gt', 'gradcam_pred', 'maskclip_gt', 'maskclip_pred', 'selfattn', 'surgery_gt', 'surgery_pred', 'm2ib_gt', 'm2ib_pred']

    hm_type = 'eclip_gt'
    print("hm type:", hm_type)

    zero_shot_weights = build_zero_shot_classifier(
        clipmodel,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True
        )
    print("[class name embeddings]:", zero_shot_weights.shape)  # [512, 1000]


    top1 = torch.zeros([11])
    top5 = torch.zeros([11])
    top10 = torch.zeros([11])
    n = torch.zeros([11])

    L = 100
    cal_gap = 10
    with open("imagenet_class_index.json", "r") as ff:
        class_dict = json.load(ff)
        for label, values in list(class_dict.items()):
            label = int(label)
            folder = values[0]
            print("Start: Processing the {}th folder, target class name: {}".format(label, IMAGENET_CLASSNAMES[label]))

            # if not os.path.exists(del_path+folder):
            #     os.makedirs(del_path+folder)
            files = os.listdir(data_path+folder)

            for f in files:
                img_name = f.split(".")[0]
                img = Image.open(os.path.join(data_path, folder, f )).convert("RGB")
                w, h = img.size
                # in case there is too large image
                if min(w,h) > 640:
                    scale = min(w,h) / 640
                    hs = int(h/scale)
                    ws = int(w/scale)
                    # print(img_name, w, h, ws, hs)
                    img = img.resize((ws,hs))
                w, h = img.size
                resize = Resize((h,w))
                # make prediction
                with torch.no_grad():
                    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
                    img_clip_embedding = clipmodel.encode_image(img_clipreprocess)
                    logits = 100. * img_clip_embedding @ zero_shot_weights
                    target = torch.tensor([label]).to(device)
                    [acc1, acc5, acc10], pred_top1 = accuracy(logits, target, topk=(1, 5, 10))
                    top1[:1] += acc1
                    top5[:1] += acc5
                    top10[:1] += acc10
                    n[0] += 1

                hm = generate_hm(hm_type, img.copy(), label, pred_top1.item(), resize)
                del_imgs = delection_process(img, hm, L, cal_gap, del_path+folder, '{}_{}'.format(img_name, hm_type)) # 10 deletion steps
                # 10, 3, 224, 224
                img_clip_embedding = clipmodel.encode_image(del_imgs)
                logits = 100. * img_clip_embedding @ zero_shot_weights
                target = torch.tensor([label]).repeat(len(img_clip_embedding)).to(device)
                [acc1, acc5, acc10], _ = accuracy(logits, target, topk=(1, 5, 10))
                top1[1:] += acc1
                top5[1:] += acc5
                top10[1:] += acc10
                n[1:] += 1

            print("[clip accuracy] Top1:")
            print("[{}]:".format(hm_type), list((top1 / n).numpy()))

            print("[clip accuracy] Top5:")
            print("[{}]:".format(hm_type), list((top5 / n).numpy()))  

            print("[clip accuracy] Top10:")
            print("[{}]:".format(hm_type), list((top10 / n).numpy()))              


    print("top1 hits:", top1)
    print("top5 hits:", top5)
    print("top10 hits:", top10)
    print("n:", n)
    top1 = top1 / n
    top5 = top5 / n
    top10 = top10 / n

    print("[clip accuracy] Top1:")
    print("[{}]:".format(hm_type), list(top1.numpy()))

    print("[clip accuracy] Top5:")
    print("[{}]:".format(hm_type), list(top5.numpy()))  

    print("[clip accuracy] Top10:")
    print("[{}]:".format(hm_type), list(top10.numpy()))       