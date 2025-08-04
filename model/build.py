from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F
import pdb

def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    nrm = (x ** 2).sum(1, keepdim=True)
    dist = nrm + nrm.T - 2.0 * (x @ x.T)
    return torch.clamp(dist, min=0.0)


def _cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return x @ x.T


def _load_clip_ckpt(ckpt_path: str, model_name: str, img_size: int, stride: int):
    """Load a fine‑tuned CLIP checkpoint *sans* optimizer/amp state."""
    model, cfg = build_CLIP_from_openai_pretrained(model_name, img_size, stride)
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model", state)
    state = {k.replace("base_model.", "") if k.startswith("base_model.") else k: v for k, v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[Teacher] loaded. missing={len(miss)}, unexpected={len(unexp)}")
    convert_weights(model)
    model.eval()
    return model, cfg, state



def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(self.embed_dim,ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(self.embed_dim,ratio=args.select_ratio)


        # 3️⃣ Teacher (optional)
        self.distill = getattr(args, "distillation", False)
        if self.distill:
            self._init_teacher(args)

 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _load_teacher_tse(self, teacher_state):
        vis_state = {k.replace("visul_emb_layer.", ""): v
                     for k, v in teacher_state.items() if k.startswith("visul_emb_layer.")}
        txt_state = {k.replace("texual_emb_layer.", ""): v
                     for k, v in teacher_state.items() if k.startswith("texual_emb_layer.")}
        self.tea_vis_emb.load_state_dict(vis_state, strict=False)
        self.tea_txt_emb.load_state_dict(txt_state, strict=False)
        print("[Teacher TSE] Visual and textual layers loaded")


    # ─────────────────────────────────────────────────────────
    # Teacher init
    # ─────────────────────────────────────────────────────────
    def _init_teacher(self, args):
        self.teacher_model, _, state_dict = _load_clip_ckpt(
            args.teacher_ckpt, args.teacher_choice, args.img_size, args.stride_size
        )
        t_dim = 768 if args.teacher_choice in {"ViT-L/14", "ViT-L/14@336px"} else 512
        self.tea_vis_emb = VisualEmbeddingLayer(ratio=args.select_ratio, input_dim=t_dim)
        self.tea_txt_emb = TexualEmbeddingLayer(ratio=args.select_ratio, input_dim=t_dim)
        self.teacher_proj = nn.Linear(t_dim, 512) if t_dim != 512 else None
        self._load_teacher_tse(state_dict)




    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()


#        pdb.set_trace()
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)


      #  pdb.set_trace()
        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 

#        pdb.set_trace()     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})


        # ----- Relational KD (optional) -----
        if self.distill:
            imgs = images
            caps = caption_ids
            stu_i_cls = i_feats
            stu_t_cls = t_feats
            stu_i_tse = i_tse_f
            stu_t_tse = t_tse_f
            with torch.no_grad():
                t_img_f, t_att_i = self.teacher_model.encode_image(imgs)
                t_txt_f, t_att_t = self.teacher_model.encode_text(caps.long())
                tea_i_cls = t_img_f[:, 0, :].float()
                tea_t_cls = t_txt_f[torch.arange(t_txt_f.size(0)), caps.argmax(-1)].float()
                tea_i_tse = self.tea_vis_emb(t_img_f, t_att_i).float()
                tea_t_tse = self.tea_txt_emb(t_txt_f, caps, t_att_t).float()

#                pdb.set_trace()
                if self.teacher_proj is not None:
                    dtype = self.teacher_proj.weight.dtype
                    tea_i_cls = self.teacher_proj(tea_i_cls.to(dtype))
                    tea_t_cls = self.teacher_proj(tea_t_cls.to(dtype))
                    tea_i_cls = tea_i_cls.float()
                    tea_t_cls = tea_t_cls.float()

            def _rkd(a: torch.Tensor, b: torch.Tensor):
                return F.mse_loss(_pairwise_distances(a), _pairwise_distances(b)) + \
                       F.mse_loss(_cosine_matrix(a), _cosine_matrix(b))

            rel_loss = (
                _rkd(stu_i_cls, tea_i_cls) +
                _rkd(stu_t_cls, tea_t_cls) 
#                _rkd(stu_i_tse, tea_i_tse) +
#                _rkd(stu_t_tse, tea_t_tse)
            )
            ret["rel_kd_loss"] = rel_loss * getattr(self.args, "rel_kd_weight", 0.1)
#            pdb.set_trace()

  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
