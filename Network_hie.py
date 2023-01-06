import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *
from .resnet import ResNet


class HHGLCM(nn.Module):

    def __init__(self, args, mode='meta',grain = 'fine'):
        super().__init__()

        self.mode = mode
        self.args = args
        self.grain = grain

        self.encoder = ResNet(args=args)

        if self.mode == 'pre_train':
            self.fc = nn.Linear(640, self.args.num_class)
        
       
    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    def get_weight_vector(self, A, B):

        M = A.shape[0]  # 75
        N = B.shape[0]  # 5
 

        A = A.unsqueeze(1) #75*1*640*5*5
        B = B.unsqueeze(0) #1*5*640*1*1

        A = A.repeat(1, N, 1, 1, 1)#75*5*640*5*5
        B = B.repeat(M, 1, 1, 1, 1)#75*5*640*5*5

        combination = (A * B).sum(2) #75*5*5*5
        combination = combination.view(M, N, -1) #75*5*25
        combination = F.relu(combination) + 1e-3
        return combination #
    
    def get_global_lobal_feature(self,x):
            gl = F.adaptive_avg_pool2d(x, 1)   # n*640*1*1
            m = x.size(-2)
            n = x.size(-1)
            mc = int(m*self.args.size_percent+1)
            nc = int(n*self.args.size_percent+1)

            lt = x[:,:,:mc,:nc]
            rt = x[:,:,-mc:,:nc]
            x[:,:,int((m-mc)/2):mc+int((m-mc)/2)+2,int((n-nc)/2):nc+int((n-nc)/2)+2]
            lb = x[:,:,:mc,-nc:]
            rb = x[:,:,-mc:,-nc:]

            local_fea = [lt,rt,mid,lb,rb]
            for i in range(len(local_fea)):
                local_fea[i] = F.adaptive_avg_pool2d(local_fea[i], 1)
            fea = torch.cat(local_fea,2)
            return gl,fea           #n*640*6*1

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)  # 5*640*5*5 

        if self.grain == 'coarse':
            proto,_ = self.get_global_lobal_feature(proto)
            query,_ = self.get_global_lobal_feature(query)
        else:
            _,proto = self.get_global_lobal_feature(proto)
            _,query = self.get_global_lobal_feature(query)

        weight_1 = self.get_weight_vector(query, proto) 
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    
    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x


    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out
