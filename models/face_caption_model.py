import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class FaceCaptionModel(nn.Module):
    def __init__(self, opt):
        super(FaceCaptionModel, self).__init__()
        self.memory_encoding_size = opt.encoding_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.classifier_type = opt.classifier_type
        self.transformer = torch.nn.Transformer(self.memory_encoding_size)

        self.face_encoding_size = opt.face_encoding_size
        self.face_feat_size = opt.face_feat_size - 6
        self.face_embed = nn.Linear(self.face_feat_size, self.face_encoding_size)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.memory_encoding_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        num_classes = opt.unique_characters + 1
        self.logit = nn.Linear(self.memory_encoding_size, num_classes)
        self.character_embed = nn.Embedding(num_classes, self.memory_encoding_size)
        self.loss = IdentificationLoss()

    def _get_required_face_features(self, slot_size, face_feats):
        face_features = None
        for slot in range(slot_size):
            # self.face_embed(face_feats[:,s,:,6:])
            face_feat = self.face_embed(face_feats[:,slot,:,6:])
            if slot > 0:
                face_features = torch.cat((face_features, face_feat),dim=1)
            else:
                face_features = face_feat

        return face_features


    def forward(self, fc_feats, face_feats, face_masks, captions, caption_masks, slots, slot_masks, slot_size,
                characters, genders=None):
        # get memory of features for each slot
        ipdb.set_trace()
        characters = characters[:,:slot_size+1]
        face_features = self._get_required_face_features(slot_size, face_feats)
        character_embed = self.character_embed(characters)
        masks = slot_masks[:,:slot_size+1].bool()

        if self.classifier_type == 'transformer':
            transformer_masks = ~masks
            src_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)-1).cuda()
            tgt_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)).cuda()
            ipdb.set_trace()
            decoder_output = self.transformer_decoder(character_embed.transpose(0,1), face_features, 
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=transformer_masks)
            logits = self.logit(decoder_output)
            logprobs = F.log_softmax(logits[:,-1,:],dim=2)


        loss = self.loss(logprobs, characters[:,1:], masks[:,1:].float())

        return loss



class IdentificationLoss(nn.Module):
    def __init__(self):
        super(IdentificationLoss, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output