import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.sent_embedding import SentEmbedding

import ipdb

class FaceCaptionModel(nn.Module):
    def __init__(self, opt):
        super(FaceCaptionModel, self).__init__()
        self.memory_encoding_size = opt.encoding_size
        self.use_bert_embedding = 'use_bert_embedding' in vars(opt) and opt.use_bert_embedding
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.classifier_type = opt.classifier_type
        self.transformer = torch.nn.Transformer(self.memory_encoding_size)

        # Video Embedding
        self.use_video = opt.use_video
        self.fc_feat_size = opt.fc_feat_size
        self.video_encoding_size = opt.video_encoding_size
        if self.use_video:
            self.fc_embed = nn.Linear(self.fc_feat_size, self.video_encoding_size)


        # Sentence Embedding
         # sent embed
        self.rnn_size = opt.rnn_size
        self.sent_embed = SentEmbedding(opt)

        # Bert Embedding 
        if self.use_bert_embedding:
            self.bert_embedding = nn.Linear(opt.bert_size, self.rnn_size)
            print('===Use Bert Embedding===', self.use_bert_embedding)

        # Face Embedding
        self.face_encoding_size = opt.face_encoding_size
        self.face_feat_size = opt.face_feat_size - 6
        self.face_embed = nn.Linear(self.face_feat_size, self.face_encoding_size)

        # Convert ViD+Caption --> memory_encoding_size
        self.final_encode = nn.Linear(self.video_encoding_size + self.rnn_size, self.memory_encoding_size)

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


    def _get_required_caption_features(self, slot_size, captions, caption_masks):
        label_features = None
        seq_len = 10
        for slot in range(slot_size):
            label_feat = self.sent_embed(captions[:,slot], caption_masks[:,slot])
            label_feat = label_feat.unsqueeze(1).repeat(1, seq_len, 1)
            if slot > 0:
                label_features = torch.cat((label_features, label_feat),dim=1)
            else:
                label_features = label_feat
        
        return label_features

    def _get_required_video_features(self, slot_size, fc_feats):
        video_features = None
        for slot in range(slot_size):
            # fc_feats[:,slot].shape --> torch.Size([64, 10, 1024])
            fc_feat = self.fc_embed(fc_feats[:,slot])
            # fc_feat.shape --> torch.Size([64, 10, 256])
            if slot > 0:
                video_features = torch.cat((video_features, fc_feat),dim=1)
            else:
                video_features = fc_feat
        
        return video_features                                                                                                                                                                                                               

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self, fc_feats, face_feats, face_masks, captions, caption_masks, slots, slot_masks, slot_size,
                characters, genders=None):
        # get memory of features for each slot
        #ipdb.set_trace()
        characters = characters[:,:slot_size+1]
        face_features = self._get_required_face_features(slot_size, face_feats)
        caption_features  = self._get_required_caption_features(slot_size, captions, caption_masks)
        video_features = self._get_required_video_features(slot_size, fc_feats)
        character_embed = self.character_embed(characters)
        masks = slot_masks[:,:slot_size+1].bool()
        ipdb.set_trace()
        vid_caption_features = torch.cat((video_features, caption_features), 2)
        vid_caption_features = self.final_encode(vid_caption_features)
        
        if self.classifier_type == 'transformer':
            transformer_masks = ~masks
            src_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)-1).cuda()
            tgt_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)).cuda()
            #ipdb.set_trace()
            cap_face_decoder_output = self.transformer_decoder(vid_caption_features.transpose(0,1), face_features.transpose(0,1))
            decoder_output = self.transformer_decoder(character_embed.transpose(0,1), cap_face_decoder_output, 
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=transformer_masks)
            logits = self.logit(self.dropout(decoder_output.transpose(0, 1)[:,:-1]))
            logprobs = F.log_softmax(logits, dim=2)


        loss = self.loss(logprobs, characters[:,1:], masks[:,1:].float())

        return loss

    def _predict(self, fc_feats, face_feats, face_masks, captions, caption_masks, slots, slot_masks, 
                 slot_size):
    
        batch_size = fc_feats.size(0)
        masks = slot_masks[:, :slot_size + 1].bool()

        face_features = self._get_required_face_features(slot_size, face_feats)
        caption_features  = self._get_required_caption_features(slot_size, captions, caption_masks)
        video_features = self._get_required_video_features(slot_size, fc_feats)
        vid_caption_features = torch.cat((video_features, caption_features), 2)

        vid_caption_features = self.final_encode(vid_caption_features)

        if self.classifier_type == 'transformer':
            transformer_masks = ~masks
            src_mask = self.transformer.generate_square_subsequent_mask(masks.size(1) - 1).cuda()
            cap_face_decoder_output = self.transformer_decoder(vid_caption_features.transpose(0,1), face_features.transpose(0,1))

            predictions = fc_feats.new_zeros(batch_size, slot_size+1, dtype=torch.long)
            for i in range(slot_size):
                character_embed = self.character_embed(predictions[:,:i+1]).transpose(0,1)
                tgt_transformer_masks = transformer_masks[:,:i+1]
                dec_output = self.transformer.decoder(character_embed, cap_face_decoder_output,
                                                tgt_mask=self.transformer.generate_square_subsequent_mask(i+1).cuda(),
                                                tgt_key_padding_mask=tgt_transformer_masks,
                                                memory_key_padding_mask=transformer_masks[:,1:]).transpose(0,1)
                dec_logit = self.logit(dec_output)[:,-1,:]
                _, tgt = torch.max(dec_logit.data, 1)
                predictions[:,i+1] = tgt
            predictions = predictions[:,1:]
        
        predicted_genders = predictions.new_zeros(predictions.size(0), predictions.size(1),dtype=torch.long)
        
        return predictions, predicted_genders


class IdentificationLoss(nn.Module):
    def __init__(self):
        super(IdentificationLoss, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output