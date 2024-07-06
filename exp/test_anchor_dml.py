import torch
from torch import optim
import dl_model.rere_tsne as tsne
import numpy as np
import dl_model.anchor_dml.anchor_dml_model as m
import dl_model.rere_config as cnf
import dl_model.dl_helper as hl
import ext.radam as radam
import dl_model.anchor_dml.anchor_sampler as sampler
import time
import pandas as pd


def train(epoch, model, optimizer, train_loader):
    return hl.train(epoch, model, optimizer, train_loader)


def param_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal(m.weight, mean=0, std=0.5)
        m.bias.data.fill_(0.01)


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


batch_size = 64
epoch_num = 500
sample_size = 200
cnf.fix_torch_random()

fp = 'G:\\anchor_dml_experiments\\i2_b64_ep100\\2022\\'
fns = ['bank_marketing', 'bank-additional', 'credit_card', 'BJ_DEPOSIT_RANKs', 'magic04',
       'spambase', 'waveform', 'credit_risk', 'diabetes', 'segment', 'HAPT', 'mnist']
# fns = ['bank_marketing']
# fns = ['renttherunway_sample']
fns = ['bank_marketing','bank-additional','BJ_DEPOSIT_RANKs','diabetes']
# fns = ['renttherunway_sample','meta_Books_vec_sample']

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
    # methods = ['hclustering','clustering', 'fa', 'ica', 'pca', 'kpca']
    methods = ['ica']
    for method in methods:
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   worker_init_fn=_init_fn)
        begin = time.time()
        fdml = fn + '_anchor_dml2_' + method
        output2 = fp + fdml + '.csv'
        img3 = fp + 'images\\' + fdml + '.png'
        model_path = fp + 'torch_models\\' + fdml

        sample_size = 2 * d_in

        sam = sampler.AnchorSampler(data_set.data.cpu().numpy(), labels, k=sample_size)
        samples = sam.getSamples(method)
        print("anchor points number:", samples.shape[0])
        samples = torch.from_numpy(samples.astype(np.float32)).to(cnf.device)
        model = m.AnchorDML(d_in, samples, d_out).to(cnf.device)
        # model.apply(param_init)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = RangerLars(model.parameters())
        inter_time = time.time()
        df = pd.DataFrame(columns=['no', 'loss', 'acc'])
        for epoch in range(1, epoch_num + 1):
            loss, acc = train(epoch, model, optimizer, train_loader)
            df.loc[epoch-1] = [epoch,loss,acc]
            # df.append({'no': epoch, 'loss': loss, 'acc': acc}, ignore_index=True)
        print(df)
        df.to_csv(fp + 'torch_models\\' + fdml + '_loss.csv', index=False)
        transform_save()
        end = time.time()
        print(fn, ' for dataset ', fn, method, ' time cost:', end - begin)
        # for p in model.named_parameters():
        #     print(p)
