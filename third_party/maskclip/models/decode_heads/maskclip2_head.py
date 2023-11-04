import torch.nn.functional as F

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@HEADS.register_module()
class MaskClip2Head(BaseDecodeHead):

    def __init__(self, img_size, **kwargs):
        super(MaskClip2Head, self).__init__(**kwargs)
        self.img_size = img_size

    def forward(self, inputs, force_output_pred_masks=False):
        assert force_output_pred_masks
        inputs_both = inputs
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        txt_embed = inputs_both[1]
        feat = inputs[-1]

        output = self.cls_seg(feat, txt_embed)

        output = F.interpolate(output, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=self.align_corners)
        output = {"pred_masks": output}

        return output

    def cls_seg(self, feat, txt_embed):
        txt_embed = txt_embed.to(feat.dtype)
        output = F.conv2d(feat, txt_embed[:, :, None, None])
        
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        raise RuntimeError('MaskClip is not trainable. Try MaskClip+ instead.')