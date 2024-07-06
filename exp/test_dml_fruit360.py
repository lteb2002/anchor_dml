import torch
from torch import optim
import dl_model.rere_tsne as tsne
import numpy as np
import time
import dl_model.rere_config as cnf
import dl_model.dl_helper as hl
import dl_model.rere_dml as dml


# from torchtools.optim import RangerLars


def train(epoch):
    hl.train(epoch, model, optimizer, train_loader)


def transform_save():
    # 变换出DML数据
    dml_data = model.transform(data_set.data.to(cnf.device)).detach().cpu().numpy()
    # 可视化DML数据
    viz2 = tsne.RereTSNE(dml_data, labels)
    viz2.save_image(img3)
    # 保存DML变换后的数据
    hl.save_numpy_data_to_csv(dml_data, labels, output2)
    # 保存训练好的模型参数
    torch.save(model.state_dict(), model_path)


batch_size = 256
epoch_num = 100
cnf.fix_torch_random()

fp = 'F:\\fruit360_exp\\'
# fns = ['fruit360_2','mind14_vec']
fns = ['mind14_vec']

for fn in fns:
    fdml = fn + '_dml_net_l2'
    input_file = fp + fn + '.csv'

    output2 = fp + fdml + '.csv'
    img1 = fp + 'images\\' + fn + '.png'
    img3 = fp + 'images\\' + fdml + '.png'
    model_path = fp + 'torch_models\\' + fdml
    begin = time.time()

    data_set = hl.ArffDataSet(input_file, normalize=True)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

    labels = data_set.labels.cpu().numpy()
    # viz = tsne.RereTSNE(data_set.data.cpu().numpy(), labels)
    # viz.save_image(img1)

    d_in = data_set.dim
    print("Dimension of the data:", d_in)
    d_out = data_set.label_num
    print("class number:", d_out)

    model = dml.RereDML(d_in, d_out).to(cnf.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = RangerLars(model.parameters())

    for epoch in range(1, epoch_num + 1):
        train(epoch)
    end = time.time()
    print(fn, ' for dataset ', fn, ' time cost:', end - begin)
    transform_save()
    # for p in model.named_parameters():
    #     print(p)
