import torch, random, os
from torch import optim
import rere_tsne as tsne
import numpy as np
import dl_model.anchor_dml.anchor_mlg_dml_model as m
import dl_model.rere_config as cnf
import dl_model.dl_helper as hl
import anchor_dml.anchor_sampler as sampler
import time
from torchviz import make_dot


def train(epoch, model, optimizer, train_loader):
    hl.train(epoch, model, optimizer, train_loader)


def _init_fn(worker_id):
    np.random.seed(int(12))


def transform_save():
    # 保存训练好的模型参数
    torch.save(model.state_dict(), model_path)
    # 变换出DML数据
    dml_data = model.encode_dml(data_set.data.to(cnf.device)).detach().cpu().numpy()
    # 可视化DML数据
    viz2 = tsne.RereTSNE(dml_data, labels)
    viz2.save_image(img3)
    # 保存DML变换后的数据
    hl.save_numpy_data_to_csv(dml_data, labels, output2)
    anchors = model.export_anchors().detach().cpu().numpy()
    als = ["anchor" for x in range(anchors.shape[0])]
    anchor_labels = np.array(als)
    hl.save_numpy_data_to_csv(anchors, anchor_labels, anchor_output)


batch_size = 64
epoch_num = 100
sample_size = 100
cnf.fix_torch_random()

fp = 'G:\\anchor_dml_experiments\\i2_b64_ep100\\'
fns = ['bank_marketing','bank-additional', 'credit_card', 'BJ_DEPOSIT_RANKs', 'magic04',
       'spambase', 'waveform', 'credit_risk','diabetes', 'segment','HAPT','mnist']
# fns = ['renttherunway_sample']
fns = ['meta_Books_vec_sample']
# fns = ['diabetes', 'segment']

for fn in fns:
    input_file = fp + fn + '.arff'
    img1 = fp + 'images\\' + fn + '.png'
    data_set = hl.ArffDataSet(input_file, normalize=True)
    labels = data_set.labels.cpu().numpy()
    viz = tsne.RereTSNE(data_set.data.cpu().numpy(), labels)
    viz.save_image(img1)
    d_in = data_set.dim
    print("Dimension of the data:", d_in)
    d_out = data_set.label_num
    print("class number:", d_out)

    # methods = ['support_vector', 'clustering','mlig']
    methods = ['mlig']
    for method in methods:
        # torch.manual_seed(12)
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   worker_init_fn=_init_fn)
        begin = time.time()
        fdml = fn + '_anchor_dml_' + method + '_l2_2m'
        output2 = fp + fdml + '.csv'
        anchor_output = fp + fn + '_anchor.csv'
        img3 = fp + 'images\\' + fdml + '.png'
        model_path = fp + 'torch_models\\' + fdml
        sam = sampler.AnchorSampler(data_set.data.cpu().numpy(), labels, k=sample_size)
        mlg = sam.getSamples(method)
        print("mlg number:", mlg.shape[0])
        mlg = torch.from_numpy(mlg.astype(np.float32)).to(cnf.device)
        # 设置PyTorch的随机种子和参数初始化方式，以实现可重复实验结果
        # torch.manual_seed(12)
        sample_size = 2*d_in
        model = m.AnchorMlgDml(d_in, mlg, sample_size, d_out).to(cnf.device)
        # for p in model.named_parameters():
        #     # print(p[0])
        #     if 'linear_trans' in p[0]:
        #         print(p)
        # model.apply(param_init)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = RangerLars(model.parameters())
        inter_time = time.time()
        for epoch in range(1, epoch_num + 1):
            train(epoch, model, optimizer, train_loader)
        transform_save()
        end = time.time()
        print(fn, ' for dataset ', fn, method, ', anchor time cost:', inter_time - begin, ', total time cost:',
              end - begin)
        # for p in model.named_parameters():
        #     if 'anchor_representation' in p[0]:
        #         print(p)
