import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import torch

# Bloom Model Output

float_path = "weights_dist/bloom_output/" # Float 
quant_path = "weights_dist/bloom_output_q/" # GPTQ per-block activation
ln_quant_path = "weights_dist/bloom_output_q_ln/"  # Norm-Tweaking per-block acitvation
# ln_quant_path = "weights_dist/bloom_output_q_ln_mean2/" #  Norm-Tweaking per-block acitvatio / per-layer learning rate

float_path_all = "weights_dist/bloom_output_q_ln_mean2_all_float/" # Float with Std
quant_path_all = "weights_dist/bloom_output_q_ln_mean2_all_gptq/" # GPTQ with Std
ln_quant_path_all = "weights_dist/bloom_output_q_ln_mean2_all/" # Float with Std

smooth_float_path = "smooth_quant_weights_dist/bloom_output_float/" # Float 
smooth_quant_path = "smooth_quant_weights_dist/bloom_output_smoothquant/" # SmoothQuant per-block activation
smooth_ln_quant_path = "smooth_quant_weights_dist/bloom_output_smoothquant_nt/"  # Norm-Tweaking per-block acitvation


def load_data():
    f_means = []
    f_stds = []

    q_means = []
    q_stds = []

    q_means_ln = []
    q_stds_ln = []

    # 30 layers
    for i in range(30):
        f_data = np.load(float_path+"layer_{}.npy".format(i))
        f_mean = np.mean(f_data[:, 0, :], axis=-1) # 128 * 2 * 4096 # B, mean/std, C
        # f_std = np.mean(f_data[:, 1, :], axis=-1)

        # f_mean, f_std = f_data[:, 0], f_data[:, 1]
        q_data = np.load(quant_path+"layer_{}.npy".format(i))
        q_mean = np.mean(q_data[:, 0, :], axis=-1)
        # q_std = np.mean(q_data[:, 1, :], axis=-1)
            
        # q_mean, q_std = q_data[:, 0], q_data[:, 1]

        q_data_ln = np.load(ln_quant_path+"layer_{}.npy".format(i))
        q_mean_ln = np.mean(q_data_ln[:, 0, :], axis=-1)
        # q_std_ln = np.mean(q_data_ln[:, 1, :], axis=-1)

        f_means.append(f_mean)
        # f_stds.append(f_std)
        q_means.append(q_mean)
        # q_stds.append(np.std(q_means))
        q_means_ln.append(q_mean_ln)
        # q_stds_ln.append(q_std_ln)

    f_means = np.array(f_means)
    # f_stds = np.array(f_stds)

    q_means = np.array(q_means)
    # q_stds = np.array(q_stds)

    q_means_ln = np.array(q_means_ln)
    # q_stds_ln = np.array(q_stds_ln)

    diff_means = f_means-q_means
    diff_means_1 = f_means-q_means_ln

    # diff_stds = np.sqrt(q_stds**2 + f_stds**2) / 100.
    # diff_stds_1 = np.sqrt(q_stds_ln**2 + f_stds**2) / 100.

    # print(diff_means.shape)

    diff_stds = np.std(diff_means, axis=-1)

    # print(diff_stds.shape)
    diff_stds_1 = np.std(diff_means_1, axis=-1)

    diff_means = np.mean(diff_means, axis=-1)
    diff_means_1 = np.mean(diff_means_1, axis=-1)

    # print(diff_means.shape)

    return diff_means, diff_stds, diff_means_1, diff_stds_1



def load_data_smooth(smooth_float_path, smooth_quant_path, smooth_ln_quant_path):
    f_means = []
    f_stds = []

    q_means = []
    q_stds = []

    q_means_ln = []
    q_stds_ln = []

    # 30 layers
    for i in range(30):
        f_data = np.load(smooth_float_path+"layer_{}.npy".format(i))
        f_mean = np.mean(f_data[:, 0, :], axis=-1) # 128 * 2 * 4096 # B, mean/std, C
        # f_std = np.mean(f_data[:, 1, :], axis=-1)

        # f_mean, f_std = f_data[:, 0], f_data[:, 1]
        q_data = np.load(smooth_quant_path+"layer_{}.npy".format(i))
        q_mean = np.mean(q_data[:, 0, :], axis=-1)
        # q_std = np.mean(q_data[:, 1, :], axis=-1)
            
        # q_mean, q_std = q_data[:, 0], q_data[:, 1]

        q_data_ln = np.load(smooth_ln_quant_path+"layer_{}.npy".format(i))
        q_mean_ln = np.mean(q_data_ln[:, 0, :], axis=-1)
        # q_std_ln = np.mean(q_data_ln[:, 1, :], axis=-1)

        f_means.append(f_mean)
        # f_stds.append(f_std)
        q_means.append(q_mean)
        # q_stds.append(np.std(q_means))
        q_means_ln.append(q_mean_ln)
        # q_stds_ln.append(q_std_ln)

    f_means = np.array(f_means)
    # f_stds = np.array(f_stds)

    q_means = np.array(q_means)
    # q_stds = np.array(q_stds)

    q_means_ln = np.array(q_means_ln)
    # q_stds_ln = np.array(q_stds_ln)

    diff_means = f_means-q_means
    diff_means_1 = f_means-q_means_ln

    # diff_stds = np.sqrt(q_stds**2 + f_stds**2) / 100.
    # diff_stds_1 = np.sqrt(q_stds_ln**2 + f_stds**2) / 100.

    # print(diff_means.shape)

    diff_stds = np.std(diff_means, axis=-1)

    # print(diff_stds.shape)
    diff_stds_1 = np.std(diff_means_1, axis=-1)

    diff_means = np.mean(diff_means, axis=-1)
    diff_means_1 = np.mean(diff_means_1, axis=-1)

    # print(diff_means.shape)

    return diff_means, diff_stds, diff_means_1, diff_stds_1


def load_data_per_channel(float_path, quant_path, ln_quant_path):
    f_means = []
    f_stds = []

    q_means = []
    q_stds = []

    q_means_ln = []
    q_stds_ln = []

    # 30 layers
    for i in range(30):
        f_data = np.load(float_path+"layer_{}.npy".format(i))
        q_data = np.load(quant_path+"layer_{}.npy".format(i))
        q_data_ln = np.load(ln_quant_path+"layer_{}.npy".format(i))

        f_means.append(f_data[:, 0, :])
        q_means.append(q_data[:, 0, :])
        q_means_ln.append(q_data_ln[:, 0, :])

    f_means = np.array(f_means)
    q_means = np.array(q_means)
    q_means_ln = np.array(q_means_ln)

    diff_means = f_means-q_means
    diff_means_1 = f_means-q_means_ln



    diff_stds = np.std(diff_means, axis=-1)
    diff_stds_1 = np.std(diff_means_1, axis=-1)
    print(diff_stds.shape)


    diff_means = np.mean(diff_means, axis=-1)
    diff_means_1 = np.mean(diff_means_1, axis=-1)
    print(diff_means.shape)


    return diff_means, diff_stds, diff_means_1, diff_stds_1


def load_data_all(float_path_all, quant_path_all, ln_quant_path_all, save_name='diff_npy/mean_std.npy'):
    # f_means = []
    # f_stds = []

    q_means = []
    q_stds = []

    q_means_ln = []
    q_stds_ln = []

    # 30 layers
    for i in tqdm(range(30)):
        f_data = np.load(float_path_all+"layer_{}.npy".format(i))
        q_data = np.load(quant_path_all+"layer_{}.npy".format(i))
        q_data_ln = np.load(ln_quant_path_all+"layer_{}.npy".format(i))

        f_torch = torch.from_numpy(f_data).cuda()
        q_torch = torch.from_numpy(q_data).cuda()
        q_torch_ln = torch.from_numpy(q_data_ln).cuda()

        gptq_diff = f_torch - q_torch
        gptq_diff_mean = torch.mean(gptq_diff)
        gptq_diff_std = torch.std(gptq_diff)

        q_means.append(gptq_diff_mean)
        q_stds.append(gptq_diff_std)

        norm_tweak_diff = f_torch - q_torch_ln
        norm_tweak_diff_mean = torch.mean(norm_tweak_diff)
        norm_tweak_diff_std = torch.std(norm_tweak_diff)
        
        q_means_ln.append(norm_tweak_diff_mean)
        q_stds_ln.append(norm_tweak_diff_std)


    diff_means = np.array([m.cpu().numpy()for m in q_means])
    diff_stds = np.array([s.cpu().numpy() for s in q_stds])

    diff_means_1 = np.array([m.cpu().numpy()for m in q_means_ln])
    diff_stds_1 = np.array([s.cpu().numpy() for s in q_stds_ln])

    np.save(save_name, [diff_means, diff_stds, diff_means_1, diff_stds_1])

    return diff_means, diff_stds, diff_means_1, diff_stds_1


def plot_mean_std_all(diff_means, diff_stds, diff_means_1, diff_stds_1):

    # # 绘制 mean 和 std 曲线
    fig, axes = plt.subplots(16,8,figsize=(7,15))
    axes = axes.flatten()

    for i in range(128):
        axes[i].plot(diff_means[:, i], label='f_mean')
        axes[i].fill_between(
            range(len(diff_means[:, i])), diff_means[:, i]-diff_stds[:, i], diff_means[:, i]+diff_stds[:, i], alpha=0.2,
        )
        axes[i].plot(diff_means_1[:, i], label='f_mean_ln')

        axes[i].fill_between(
            range(len(diff_means_1[:, i])), diff_means_1[:, i]-diff_stds_1[:, i], diff_means_1[:, i]+diff_stds_1[:, i], alpha=0.2,
        )

        axes[i].axis('off')
        # plt.plot(q_means[:, i], label='q_mean')
        # plt.fill_between(
        #     range(len(q_means[:, i])), q_means[:, i]-q_stds[:, i], q_means[:, i]+q_stds[:, i], alpha=0.2,
        # )
        # axes[i].legend()
    plt.tight_layout()
    plt.savefig("plot_imgs_bo/seq_all_channel-diff_mean_std_ln_per_layer_lr.png")
        # break
        # plt.clf()



def plot_mean_std_single(diff_means, diff_stds, diff_means_1, diff_stds_1):

    # # 绘制 mean 和 std 曲线
    fig, axes = plt.subplots(2,4,figsize=(5,3))
    axes = axes.flatten()
    # axes = [axes]

    selected = [119, 118, 117, 105, 101, 99, 95, 24]
    # for i in range(128):
    for j, i in enumerate(selected):
        axes[j].plot(diff_means[:, i], label='f_mean')
        # axes[j].fill_between(
        #     range(len(diff_means[:, i])), diff_means[:, i]-diff_stds[:, i], diff_means[:, i]+diff_stds[:, i], alpha=0.2,
        # )
        axes[j].plot(diff_means_1[:, i], label='f_mean_ln')

        # axes[j].fill_between(
        #     range(len(diff_means_1[:, i])), diff_means_1[:, i]-diff_stds_1[:, i], diff_means_1[:, i]+diff_stds_1[:, i], alpha=0.2,
        # )

        # break
        axes[j].axis('off')

        
    plt.tight_layout()
    plt.savefig("plot_imgs_bo/selected_diff_std.png")


def plot_mean_rel():
    # 119, 118, 117, 105, 101, 99, 95, 24

    f_means = np.mean(f_means, axis=1)
    q_means = np.mean(q_means, axis=1)
    f_stds = np.mean(f_stds, axis=1)
    q_stds = np.mean(q_stds, axis=1)

    diff_means = np.abs(q_means-f_means)
    diff_stds = np.sqrt(q_stds**2 + f_stds**2)

    plt.plot(f_means, label='f_mean')
    plt.fill_between(
    range(len(f_means)), f_means-f_stds, f_means+f_stds, alpha=0.2,
    )
    plt.plot(q_means, label='q_mean')
    plt.fill_between(
    range(len(q_means)), q_means-q_stds, q_means+q_stds, alpha=0.2,
    )
    plt.plot(diff_means, label='diff_mean')
    plt.fill_between(
    range(len(diff_means)), diff_means-diff_stds, diff_means+diff_stds, alpha=0.2,
    )
    plt.legend()
    plt.savefig("plot_imgs_bo/seq_mean_rel.png")
    plt.clf()



def plot_mean_std_all_in_one(diff_means, diff_stds, diff_means_1, diff_stds_1, show_std=False):

    # # 绘制 mean 和 std 曲线
    fig, axes = plt.subplots(1,1,figsize=(5,3))
    # plt.rcParams['font.family'] = 'DejaVu Sans'

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rcParams["patch.force_edgecolor"] = True
    # axes = axes.flatten()
    axes = [axes]

    axes[0].plot(diff_means, label='GPTQ')

    if show_std:
        axes[0].fill_between(
            range(len(diff_means)), diff_means-diff_stds, diff_means+diff_stds, alpha=0.2,
        )
    axes[0].plot(diff_means_1, label='Norm Tweaking')

    if show_std:
        axes[0].fill_between(
            range(len(diff_means_1)), diff_means_1-diff_stds_1, diff_means_1+diff_stds_1, alpha=0.2,
        )

    axes[0].set_ylabel("$\Delta_\mu$", rotation=0, labelpad=10)
    axes[0].set_xlabel("Layer")

    axes[0].legend(loc="upper left")

    plt.tight_layout()
    if show_std:
        save_name = "plot_imgs_bo/seq_mean_all_in_one_old_show_std.pdf"
    else:
        save_name = "plot_imgs_bo/seq_mean_all_in_one_old.pdf"

    plt.savefig(save_name)


def plot_mean_std_all_in_one_pair(diff_means, diff_stds, diff_means_1, diff_stds_1,
                                diff_means_smooth, diff_stds_smooth, diff_means_1_smooth, diff_stds_1_smooth,
                                show_std=True):
    # # 绘制 mean 和 std 曲线
    fig, axes = plt.subplots(1,2,figsize=(5,3))
    # plt.rcParams['font.family'] = 'DejaVu Sans'

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rcParams["patch.force_edgecolor"] = True
    axes = axes.flatten()
    # axes = [axes]

    # GPTQ
    axes[0].plot(diff_means, label='GPTQ')

    if show_std:
        axes[0].fill_between(
            range(len(diff_means)), diff_means-diff_stds, diff_means+diff_stds, alpha=0.2,
        )
    axes[0].plot(diff_means_1, label='Norm Tweaking')

    if show_std:
        axes[0].fill_between(
            range(len(diff_means_1)), diff_means_1-diff_stds_1, diff_means_1+diff_stds_1, alpha=0.2,
        )

    axes[0].set_ylabel("$\Delta_\mu$", rotation=0, labelpad=10)
    axes[0].set_xlabel("Layer")
    axes[0].legend()


    # SmoothQuant
    axes[1].plot(diff_means_smooth, label='SmoothQuant')

    if show_std:
        axes[1].fill_between(
            range(len(diff_means_smooth)), diff_means_smooth-diff_stds_smooth, diff_means_smooth+diff_stds_smooth, alpha=0.2,
        )
    axes[1].plot(diff_means_1_smooth, label='Norm Tweaking')

    if show_std:
        axes[1].fill_between(
            range(len(diff_means_1_smooth)), diff_means_1_smooth-diff_stds_1_smooth, diff_means_1_smooth+diff_stds_1_smooth, alpha=0.2,
        )

    axes[1].set_ylabel("$\Delta_\mu$", rotation=0, labelpad=10)
    axes[1].set_xlabel("Layer")
    axes[1].legend()



    plt.tight_layout()
    if show_std:
        save_name = "plot_imgs_bo/seq_mean_all_in_one_old_show_std_smooth.pdf"
    else:
        save_name = "plot_imgs_bo/seq_mean_all_in_one_old_smooth.pdf"

    plt.savefig(save_name)

def main():
    # bs=128
    diff_means, diff_stds, diff_means_1, diff_stds_1 = load_data()
    # plot_mean_std_all_in_one(diff_means, diff_stds, diff_means_1, diff_stds_1, show_std=True)

    # diff_means_smooth, diff_stds_smooth, diff_means_1_smooth, diff_stds_1_smooth = load_data_smooth(smooth_float_path, smooth_quant_path, smooth_ln_quant_path)

    diff_means_smooth, diff_stds_smooth, diff_means_1_smooth, diff_stds_1_smooth = np.load('diff_npy/mean_std_smooth.npy')

    plot_mean_std_all_in_one_pair(diff_means, diff_stds, diff_means_1, diff_stds_1,
                                  diff_means_smooth, diff_stds_smooth, diff_means_1_smooth, diff_stds_1_smooth,
                                  show_std=False)

    # load_data_all(smooth_float_path, smooth_quant_path, smooth_ln_quant_path, 'diff_npy/mean_std_smooth.npy')

    # plot_mean_std_all(diff_means, diff_stds, diff_stds, diff_stds_1)


    # diff_means, diff_stds, diff_stds, diff_stds_1 = load_data_per_channel()
    # plot_mean_std_single(diff_means, diff_stds, diff_stds, diff_stds_1)


    # diff_means, diff_stds, diff_stds, diff_stds_1 = load_data_all()
    # diff_means, diff_stds, diff_stds, diff_stds_1 = np.load('diff_npy/mean_std.npy')
    # print(diff_means)
    # plot_mean_std_all_in_one(diff_means, diff_stds, diff_stds, diff_stds_1)
    

if __name__ == "__main__":
    main()