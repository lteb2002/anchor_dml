import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.ticker import FormatStrFormatter
import cluster_analysis.cluster_evaluation as cl_eval
import time

file_path = r"G:\anchor_dml_experiments\i2_b64_ep100\2022\pics\\"
# fns = ["bank_marketing_loss", 'bank-additional_loss', 'credit_card_loss','credit_risk_loss',
#        'diabetes_loss','magic04_loss','segment_loss','spambase_loss','waveform_loss','BJ_DEPOSIT_RANKs_loss','meta_Books_vec_sample_loss','renttherunway_sample_loss']

fns = ['fruit360_loss','mind14_loss']

for fn in fns:
    fp = file_path + fn + ".csv"
    pgn = file_path + fn + "_loss_acc.png"
    data = pd.read_csv(fp)

    losses = data['loss'].tolist()
    accs = data['acc'].tolist()
    xt = range(1, len(losses) + 1)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 3)
    # ax1.xticks(xt)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.set_xticks(xt)
    ln1 = ax1.plot(xt, losses, lw=1, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color='tab:red')
    # ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy', color='tab:blue')  # we already handled the x-label with ax1
    ln2 = ax2.plot(xt, accs, lw=1, color='tab:blue', linestyle='--', label='Acc.')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax2.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(ln1+ln2,['Loss','Acc.'],loc='right')
    # plt.legend(loc='upper right')
    plt.savefig(pgn)
    # plt.show()
