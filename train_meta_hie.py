import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import HGLCM
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time 

PRETRAIN_DIR='pre_train/'
DATA_DIR='datas/'

parser = argparse.ArgumentParser()
#about dataset and training
parser.add_argument('-dataset', type=str, default='fc100', choices=['tieredimagenet','fc100','cifar_fs'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
parser.add_argument('-set',type=str,default='val',choices=['test','val'],help='the set used for validation')# set used for validation
#about training
parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.00001)
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument("-c","--size_percent",type = float, default=1/2)
parser.add_argument('-val_frequency',type=int,default=50)
parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch')
parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
#about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15,help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=1000, help='number of validation episode')
parser.add_argument('-test_episode', type=int, default=5000, help='number of testing episodes after training')
# about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn'])
#deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
#deepemd sampling only
parser.add_argument('-num_patch',type=int,default=9)
#deepemd grid only patch_list
parser.add_argument('-patch_list',type=str,default='2,3',help='the size of grids at every image-pyramid level')
parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
# slvoer about
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
parser.add_argument('-form', type=str, default='L2', choices=['L2'])
parser.add_argument('-l2_strength', type=float, default=0.000001)
# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=10, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

# OTHERS
parser.add_argument('-gpu', default='1')
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)

args = parser.parse_args()
pprint(vars(args))

#transform str parameter into list
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)

# model
args.pretrain_dir=osp.join(args.pretrain_dir,'%s/128-0.1000-30-0.20/max_acc.pth'%(args.dataset))
model = HGLCM(args)
model = load_model(model, args.pretrain_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()


args.save_path = '%s/%s/%dshot-%dway/'%(args.dataset,args.deepemd,args.shot,args.way)

args.save_path=osp.join('hie_loss_gl/',args.save_path)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)


trainset = Dataset('train', args)
train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

valset = Dataset(args.set, args)
val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

if not args.random_val_task:
    print ('fix val set for all epochs')
    val_loader=[x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

#label for query set, always in the same pattern
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)#012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()



optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

def save_model(name):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))


def tree_ANcestor(tr, nd):
   #  tree: 二维数组   node：节点  默认根节点为0

   A = [nd]  # 存储
   nd_ = tr[nd - 1]   # python 从 0 开始算，而结点从 1 开始算

   while nd_ > 0:
      A.append(nd_)
      nd_ = tr[nd_ - 1]   # 找父结点

   return A

# TIE
def EvaHier_TreeInducedError(tr, p_nd, r_nd):

    TIE = 0
    for i in range(len(p_nd)):
        r_anc = tree_ANcestor(tr, r_nd[i])   # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])   # 预测标签的父结点
        b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
        c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
        TIE = TIE + len(b + c)

    TIE = TIE/len(p_nd)  
    return TIE

# FH
def EvaHier_HierarchicalPrecisionAndRecall(tr, p_nd, r_nd):
   
   sum_PH, sum_RH, sum_FH = 0, 0, 0
   length = len(p_nd)
   for i in range(length):
      r_anc = tree_ANcestor(tr, r_nd[i])   # 真实标签的父结点
      p_anc = tree_ANcestor(tr, p_nd[i])   # 预测标签的父结点
      b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集

      PH = len(b) / len(p_anc)
      RH = len(b) / len(r_anc)
      FH = 2 * PH * RH / (PH + RH)

      sum_PH = sum_PH + PH
      sum_RH = sum_RH + RH
      sum_FH = sum_FH + FH
   
   PH = sum_PH / length
   RH = sum_RH / length
   FH = sum_FH / length

   return FH  #, PH, RH

# create tree_array
def create_array(coarse_list):

   num_coarse = max(coarse_list) + 1
   tree = [0]
   for i in range(num_coarse):
      tree.append(1)

   for x in coarse_list:
      tree.append(x+2)

   return tree

def get_coarse_label(label):
    
    c_label = []
    for i in range(args.way):
        if label[i].item() in c_label:
            continue
        else:
            c_label.append(label[i].item())
    c_label = dict(zip(c_label, range(len(c_label))))
    coarse_label = torch.arange(label.size(0))
    for i in range(label.size(0)):
        coarse_label[i] = c_label[label[i].item()]
    return coarse_label.cuda()
    
def get_sample_parent(sample_coarse_labels):
    sample_parents = {}  # 整合到父节点的数组下
    for i in range(sample_coarse_labels.size(0)):
        if sample_coarse_labels[i].item() not in sample_parents.keys():
            sample_parents[sample_coarse_labels[i].item()] = [i]
        else:
            sample_parents[sample_coarse_labels[i].item()].append(i)
    return sample_parents

trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path,'tf'))

result_list=[args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print (args.save_path)
    start_time=time.time()

    tl = Averager()
    ta = Averager()


    tqdm_gen = tqdm.tqdm(train_loader)
    # lr_scheduler.step()
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm_gen, 1):

        global_count = global_count + 1
        data, _, coarse_label = [_ for _ in batch]
        coarse_label = get_coarse_label(coarse_label)

        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data.cuda())
        data_shot, data_query = data[:k], data[k:]   # 5*640*13*1
        coarse_shot,coarse_query = coarse_label[:k], coarse_label[k:]
    
        model.module.mode = 'meta'
        if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
                coarse_shot = coarse_shot[:args.shot]
        sample_parents = get_sample_parent(coarse_shot)
       
        for i, (key,value) in enumerate(sample_parents.items()):
            sample_coarse_ext = data_shot[value,:].mean(dim=0).unsqueeze(0)  # 获取每个父类特征的平均值
            if i == 0:
                sample_coarse_exts = sample_coarse_ext
            else:
                sample_coarse_exts = torch.cat((sample_coarse_exts,sample_coarse_ext),0) 

        # model.module.mode = 'meta'
        model.module.grain = 'coarse'
        coarse_logits = model((sample_coarse_exts.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss_c = F.cross_entropy(coarse_logits, coarse_query)

        model.module.grain = 'fine'
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss = F.cross_entropy(logits, label)

        all_logits = torch.zeros(data_query.size(0), args.way).cuda()
        for i,(key,value) in enumerate(sample_parents.items()):
            for j in value:
                all_logits[:,j] = 0.6*logits[:,j] + 0.4*coarse_logits[:,i]

        pred = torch.argmax(all_logits, dim=1)
        loss_f = 0
        
        for j in range(data_query.size(0)):
            loss0 = F.cross_entropy(logits[j,:].view(1,-1), label[j].reshape((1,)))
            if coarse_shot[pred[j].item()] == coarse_query[j]:
                if pred[j] == label[j]:
                    w = 0
                else:
                    w = 1/2
            else:
                w = 1
            loss_f += (1+w)*loss0
        loss_f = loss_f/data_query.size(0)

        loss = loss_f 
        acc_f = count_acc(logits, label)
        acc_c = count_acc(coarse_logits, coarse_query)
        acc = count_acc(logits, label)
        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)

        total_loss = loss/args.bs#batch of tasks, done by accumulate gradients
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc_f={:.4f} acc_c={:.4f} acc={:.4f}'
              .format(epoch, total_loss.item(), acc_f,acc_c,acc))
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        detect_grad_nan(model)
        if i%args.bs==0: #batch of tasks, done by accumulate gradients
            optimizer.step()
            optimizer.zero_grad()


    tl = tl.item()
    ta = ta.item()
    vl = Averager()
    va = Averager()
    va_f = Averager()
    va_c = Averager()

    #validation
    model.eval()
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            global_count = global_count + 1
            data, _, coarse_label = [_ for _ in batch]
            coarse_label = get_coarse_label(coarse_label)

            k = args.way * args.shot
            model.module.mode = 'encoder'
            data = model(data.cuda())
            data_shot, data_query = data[:k], data[k:]   # 5*640*13*1
            coarse_shot,coarse_query = coarse_label[:k], coarse_label[k:]
            model.module.mode = 'meta'
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
                coarse_shot = coarse_shot[:args.shot]
            sample_parents = get_sample_parent(coarse_shot)
        
            for i, (key,value) in enumerate(sample_parents.items()):
                sample_coarse_ext = data_shot[value,:].mean(dim=0).unsqueeze(0)  # 获取每个父类特征的平均值
                if i == 0:
                    sample_coarse_exts = sample_coarse_ext
                else:
                    sample_coarse_exts = torch.cat((sample_coarse_exts,sample_coarse_ext),0) 

            # model.module.mode = 'meta'
            model.module.grain = 'coarse'
            coarse_logits = model((sample_coarse_exts.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            loss_c = F.cross_entropy(coarse_logits, coarse_query)

            
            model.module.grain = 'fine'
            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits, label)

            all_logits = torch.zeros(data_query.size(0), args.way).cuda()
            for i,(key,value) in enumerate(sample_parents.items()):
                for j in value:
                    all_logits[:,j] = 0.6*logits[:,j] + 0.4*coarse_logits[:,i]

            pred = torch.argmax(all_logits, dim=1)
            loss_f = 0
            
            for j in range(data_query.size(0)):
                loss0 = F.cross_entropy(logits[j,:].view(1,-1), label[j].reshape((1,)))
                if coarse_shot[pred[j].item()] == coarse_query[j]:
                    if pred[j] == label[j]:
                        w = 0
                    else:
                        w = 1/2
                else:
                    w = 1
                loss_f += (1+w)*loss0
            loss_f = loss_f/data_query.size(0)

            loss = loss_f
          
            acc_f = count_acc(logits, label)
            acc_c = count_acc(coarse_logits, coarse_query)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)
            va_f.add(acc_f)
            va_c.add(acc_c)

    vl = vl.item()
    va = va.item()
    va_f = va_f.item()
    va_c = va_c.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    writer.add_scalar('data/val_f_acc', float(va_f), epoch)
    writer.add_scalar('data/val_c_acc', float(va_c), epoch)
    tqdm_gen.set_description('epo {}, val, loss={:.4f} acc_f={:.4f} acc_c={:.4f} acc={:.4f}'.format(epoch, vl, va_f,va_c,va))

    print ('val_f acc:%.4f val_c acc:%.4f val acc:%.4f '%(va_f,va_c,va))
    if va >= trlog['max_acc']:
        print ('*********A better model is found*********')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    result_list.append('epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f'%(epoch,tl,ta,vl,va))

    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all:
        save_model('epoch-%d'%epoch)
        torch.save(optimizer.state_dict(), osp.join(args.save_path,'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print ('This epoch takes %d seconds'%(time.time()-start_time),'\nstill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
    lr_scheduler.step()

writer.close()



# Test Phase
trlog = torch.load(osp.join(args.save_path, 'trlog'))
test_set = Dataset('test', args)
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
test_acc_record = np.zeros((args.test_episode,))
test_tie_record = np.zeros((args.test_episode,))
test_fh_record = np.zeros((args.test_episode,))
test_f_acc_record = np.zeros((args.test_episode,))
test_c_acc_record = np.zeros((args.test_episode,))
model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
print("load",osp.join(args.save_path, 'max_acc' + '.pth'))
model.eval()

ave_acc = Averager()
ave_tie = Averager()
ave_fh = Averager()
ave_f_acc = Averager()
ave_c_acc = Averager()

label = torch.arange(args.way).repeat(args.query)
if torch.cuda.is_available():
    label = label.type(torch.cuda.LongTensor)
else:
    label = label.type(torch.LongTensor)

tqdm_gen = tqdm.tqdm(loader)
with torch.no_grad():
    for ind, batch in enumerate(tqdm_gen, 1):
        data, _, coarse_label = [_ for _ in batch]
        coarse_label = get_coarse_label(coarse_label)

        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data.cuda())
        data_shot, data_query = data[:k], data[k:]   # 5*640*13*1
        coarse_shot,coarse_query = coarse_label[:k], coarse_label[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
            coarse_shot = coarse_shot[:args.shot]
        sample_parents = get_sample_parent(coarse_shot)
    
        for i, (key,value) in enumerate(sample_parents.items()):
            sample_coarse_ext = data_shot[value,:].mean(dim=0).unsqueeze(0)  # 获取每个父类特征的平均值
            if i == 0:
                sample_coarse_exts = sample_coarse_ext
            else:
                sample_coarse_exts = torch.cat((sample_coarse_exts,sample_coarse_ext),0) 

        # model.module.mode = 'meta'
        model.module.grain = 'coarse'
        coarse_logits = model((sample_coarse_exts.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss_c = F.cross_entropy(coarse_logits, coarse_query)

        
        model.module.grain = 'fine'
        logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
        loss = F.cross_entropy(logits, label)

        all_logits = torch.zeros(data_query.size(0), args.way).cuda()
        for i,(key,value) in enumerate(sample_parents.items()):
            for j in value:
                all_logits[:,j] = 0.6*logits[:,j] + 0.4*coarse_logits[:,i]
        pred = torch.argmax(all_logits, dim=1)
       
        loss_f = 0
        
        for j in range(data_query.size(0)):
            loss0 = F.cross_entropy(logits[j,:].view(1,-1), label[j].reshape((1,)))
            if coarse_shot[pred[j].item()] == coarse_query[j]:
                if pred[j] == label[j]:
                    w = 0
                else:
                    w = 1/2
            else:
                w = 1
            loss_f += (1+w)*loss0
        loss_f = loss_f/data_query.size(0)

        loss = loss_f 

        acc_f = count_acc(logits, label)*100
        acc_c = count_acc(coarse_logits, coarse_query)*100
        acc = count_acc(logits, label)*100
        tr = create_array(coarse_shot)
        coarse_num = len(torch.unique(coarse_shot))
        tie = EvaHier_TreeInducedError(tr, pred+2+coarse_num, label+2+coarse_num)
        fh = EvaHier_HierarchicalPrecisionAndRecall(tr, pred+2+coarse_num, label+2+coarse_num)

        ave_f_acc.add(acc_f)
        ave_f_acc.add(acc_c)
        ave_acc.add(acc)
        ave_tie.add(tie)
        ave_fh.add(fh)
        
        test_acc_record[ind-1] = acc
        test_tie_record[ind - 1] = tie
        test_fh_record[ind - 1] = fh
        test_f_acc_record[ind-1] = acc_f
        test_c_acc_record[ind-1] = acc_c
        tqdm_gen.set_description('batch {}: {:.2f}({:.2f})'.format(k, ave_acc.item(), acc))


m, pm = compute_confidence_interval(test_acc_record)
m_tie, pm_tie = compute_confidence_interval(test_tie_record)
m_fh, pm_fh = compute_confidence_interval(test_fh_record)
m_f, pm_f = compute_confidence_interval(test_f_acc_record)
m_c, pm_c = compute_confidence_interval(test_c_acc_record)

result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}, \nbest test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
result_list.append('Test Acc {:.4f} + {:.4f} acc_tie {:.4f} + {:.4f} acc_fh {:.4f} + {:.4f} f_acc {:.4f} + {:.4f}  c_acc {:.4f} + {:.4f}'.format(m, pm,m_tie, pm_tie,m_fh, pm_fh,m_f, pm_f,m_c,pm_c))
print (result_list[-2])
print (result_list[-1])
save_list_to_txt(os.path.join(args.save_path,'results.txt'),result_list)
