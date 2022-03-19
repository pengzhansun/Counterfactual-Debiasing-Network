import torch
import torch.nn as nn
from model.resnet3d_xl import Net
from torchvision.ops.roi_align import roi_align


def box_to_normalized(boxes_tensor, crop_size=[224, 224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[..., 0] = (
                                       boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 1] = (
                                       boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0) * crop_size[1]
    new_boxes_tensor[..., 2] = (
                                       boxes_tensor[..., 0] + boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 3] = (
                                       boxes_tensor[..., 1] + boxes_tensor[..., 3] / 2.0) * crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor


def build_region_feas(feature_maps, boxes_list, output_crop_size=[3, 3], img_size=[224, 224]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps, boxes_list, output_crop_size,
                            spatial_scale=float(FW) / IW)  # b*T*K, C, S, S; S denotes output_size
    return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S


class VideoGlobalModel(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, opt,
                 ):
        super(VideoGlobalModel, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        self.fc = nn.Linear(512, self.nr_actions)
        self.crit = nn.CrossEntropyLoss()

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """

        # org_features - [V x 2048 x T / 2 x 14 x 14]
        org_features = self.i3D(global_img_input)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)

        # Get global features - [V x 512]
        global_features = self.avgpool(videos_features).squeeze()
        global_features = self.dropout(global_features)

        cls_output = self.fc(global_features)
        return cls_output, global_features


class BboxVisualModel(nn.Module):
    '''
    backbone: i3d
    '''

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.nr_boxes = opt.num_boxes

        self.img_feature_dim = 512
        self.backbone = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.conv = nn.Conv2d(2048, self.img_feature_dim, kernel_size=(1, 1), stride=1)
        self.crop_size = [3, 3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size[0] * self.crop_size[1], self.img_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.gru = nn.GRU(input_size=self.img_feature_dim,
                          hidden_size=self.img_feature_dim,
                          num_layers=1,
                          batch_first=True)

        self.aggregate_func = nn.Sequential(
            nn.Linear(self.nr_boxes * self.img_feature_dim, self.img_feature_dim),
            nn.BatchNorm1d(self.img_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.img_feature_dim, self.img_feature_dim),
            nn.BatchNorm1d(self.img_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.img_feature_dim, self.img_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.img_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        self.fc = nn.Linear(512, self.nr_actions)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):

        '''
        b, _, T, H, W  = global_img_input.size()
        global_img_input = global_img_input.permute(0, 2, 1, 3, 4).contiguous()
        global_img_input = global_img_input.view(b*T, 3, H, W)
        org_feas = self.backbone(global_img_input) # (b*T, 2048)
        conv_fea_maps = self.conv(org_feas)  # (b*T, img_feature_dim)
        box_tensors = box_input.view(b * T, self.nr_boxes, 4)
        '''

        org_feas = self.backbone(global_img_input)  # (b, 2048, T/2, 14, 14)
        b, _, T, H, W = org_feas.size()
        org_feas = org_feas.permute(0, 2, 1, 3, 4).contiguous()
        org_feas = org_feas.view(b * T, 2048, H, W)
        conv_fea_maps = self.conv(org_feas)  # (b*T, img_feature_dim)
        box_tensors = box_input.view(b * T, self.nr_boxes, 4)

        boxes_list = box_to_normalized(box_tensors, crop_size=[224, 224])
        img_size = global_img_input.size()[-2:]

        # (b*T*nr_boxes, C), C=3*3*d
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size, img_size)

        region_vis_feas = self.region_vis_embed(region_vis_feas)

        region_vis_feas = region_vis_feas.view(b, T, self.nr_boxes,
                                               region_vis_feas.size(-1))  # (b, t, n, img_feature_dim)
        region_vis_feas = region_vis_feas.permute(0, 3, 2, 1).contiguous()

        global_features = self.avgpool(region_vis_feas).squeeze()
        global_features = self.dropout(global_features)
        cls_output = self.fc(global_features)
        return cls_output, global_features


class BboxInteractionLatentModel(nn.Module):
    '''
    Add bbox category embedding
    '''

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.coord_feature_dim = opt.coord_feature_dim

        self.interaction = nn.Sequential(
            nn.Linear(self.nr_boxes * 4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        # Fusion of Object Interaction and Category Embedding
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.coord_feature_dim + self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )

        self.temporal_aggregate_func = nn.Sequential(
            nn.Linear(self.nr_frames * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.object_compose_func = nn.Sequential(
            nn.Linear(self.nr_boxes * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        b, _, _, _ = box_input.size()  # (b, T, nr_boxes, 4)

        box_categories = box_categories.long()

        box_categories = box_categories.view(b * self.nr_frames * self.nr_boxes)
        box_category_embeddings = self.category_embed_layer(box_categories)
        identity_repre = box_category_embeddings.view(b, self.nr_frames, self.nr_boxes,
                                                      -1)  # (b, t, n, coord_feature_dim//2)

        # Calculate the distance vector between objects
        box_dis_vec = box_input.unsqueeze(3) - box_input.unsqueeze(2)  # (b, T, nr_boxes, nr_boxes, 4)

        box_dis_vec_inp = box_dis_vec.view(b * self.nr_frames * self.nr_boxes, -1)
        inter_fe = self.interaction(box_dis_vec_inp)
        inter_fe = inter_fe.view(b, self.nr_frames, self.nr_boxes, -1)  # (b, T, nr_boxes, coord_feature_dim)

        inter_fea_latent = torch.cat([inter_fe, identity_repre], dim=-1)  # (b, T, nr_boxes, dim+dim//2)
        inter_fea_latent = inter_fea_latent.view(-1, inter_fea_latent.size()[-1])

        inter_fe = self.fuse_layer(inter_fea_latent)
        inter_fe = inter_fe.view(b, self.nr_frames, self.nr_boxes, -1).transpose(2, 1).contiguous()  # (b, nr_boxes, T, coord_feture_dim)

        inter_fea_inp = inter_fe.view(b * self.nr_boxes, -1)
        obj_inter_fea = self.temporal_aggregate_func(inter_fea_inp)
        obj_inter_fea = obj_inter_fea.view(b, self.nr_boxes, -1)  # (b, nr_boxes, coord_feature_dim)

        obj_fe = obj_inter_fea

        obj_inter_fea_inp = obj_inter_fea.view(b, -1)
        video_fe = self.object_compose_func(obj_inter_fea_inp)

        cls_output = self.classifier(video_fe)

        return cls_output, video_fe


class ConcatFusionModel(nn.Module):
    '''
    Input: the vision feature extracted from i3d backbone and the coord feature extracted from coord branch
    '''
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.coord_feature_dim = 512
        self.img_feature_dim = 512
        self.fusion_feature_dim = self.img_feature_dim + self.coord_feature_dim
        self.nr_actions = opt.num_classes

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, self.fusion_feature_dim, bias=False),
            nn.BatchNorm1d(self.fusion_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_feature_dim, self.fusion_feature_dim, bias=False),
            nn.BatchNorm1d(self.fusion_feature_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, self.fusion_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, feature_vision, feature_coord):
        concat_feature = torch.cat([feature_vision, feature_coord], -1)
        fusion_feature = self.fusion(concat_feature)
        cls_output = self.classifier(fusion_feature)
        return cls_output

