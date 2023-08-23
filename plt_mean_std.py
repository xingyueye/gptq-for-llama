import numpy as np
import matplotlib.pyplot as plt

float_path = "weights_dist/bloom_output/"
quant_path = "weights_dist/bloom_output_q/"
ln_quant_path = "weights_dist/bloom_output_q_ln/"
f_means = []
f_stds = []
q_means = []
q_stds = []
q_means_ln = []
q_stds_ln = []
for i in range(30):
    f_data = np.load(float_path+"layer_{}.npy".format(i))
    f_mean = np.mean(f_data[:, 0, :], axis=-1)
    f_std = np.mean(f_data[:, 1, :], axis=-1)
    # f_mean, f_std = f_data[:, 0], f_data[:, 1]
    q_data = np.load(quant_path+"layer_{}.npy".format(i))
    q_mean = np.mean(q_data[:, 0, :], axis=-1)
    q_std = np.mean(q_data[:, 1, :], axis=-1)
    # q_mean, q_std = q_data[:, 0], q_data[:, 1]

    q_data_ln = np.load(ln_quant_path+"layer_{}.npy".format(i))
    q_mean_ln = np.mean(q_data_ln[:, 0, :], axis=-1)
    q_std_ln = np.mean(q_data_ln[:, 1, :], axis=-1)

    f_means.append(f_mean)
    f_stds.append(f_std)
    q_means.append(q_mean)
    q_stds.append(q_std)
    q_means_ln.append(q_mean_ln)
    q_stds_ln.append(q_std_ln)
f_means = np.array(f_means)
f_stds = np.array(f_stds)

q_means = np.array(q_means)
q_stds = np.array(q_stds)

q_means_ln = np.array(q_means_ln)
q_stds_ln = np.array(q_stds_ln)

diff_means = np.abs(q_means-f_means)
diff_means_1 = np.abs(q_means_ln-f_means)

diff_stds = np.sqrt(q_stds**2 + f_stds**2) / 100.
diff_stds_1 = np.sqrt(q_stds_ln**2 + f_stds**2) / 100.

# # 绘制 mean 和 std 曲线
for i in range(128):
    plt.plot(diff_means[:, i], label='f_mean')
    plt.fill_between(
        range(len(diff_means[:, i])), diff_means[:, i]-diff_stds[:, i], diff_means[:, i]+diff_stds[:, i], alpha=0.2,
    )
    plt.plot(diff_means_1[:, i], label='f_mean_ln')
    plt.fill_between(
        range(len(diff_means_1[:, i])), diff_means_1[:, i]-diff_stds_1[:, i], diff_means_1[:, i]+diff_stds_1[:, i], alpha=0.2,
    )
    # plt.plot(q_means[:, i], label='q_mean')
    # plt.fill_between(
    #     range(len(q_means[:, i])), q_means[:, i]-q_stds[:, i], q_means[:, i]+q_stds[:, i], alpha=0.2,
    # )
    plt.legend()
    plt.savefig("plot_imgs/seq_{}_channel-diff_mean_std_ln.png".format(i))
    plt.clf()

# f_means = np.mean(f_means, axis=1)
# q_means = np.mean(q_means, axis=1)
# f_stds = np.mean(f_stds, axis=1)
# q_stds = np.mean(q_stds, axis=1)

# diff_means = np.abs(q_means-f_means)
# diff_stds = np.sqrt(q_stds**2 + f_stds**2)

# plt.plot(f_means, label='f_mean')
# plt.fill_between(
# range(len(f_means)), f_means-f_stds, f_means+f_stds, alpha=0.2,
# )
# plt.plot(q_means, label='q_mean')
# plt.fill_between(
# range(len(q_means)), q_means-q_stds, q_means+q_stds, alpha=0.2,
# )
# plt.plot(diff_means, label='diff_mean')
# plt.fill_between(
# range(len(diff_means)), diff_means-diff_stds, diff_means+diff_stds, alpha=0.2,
# )
# plt.legend()
# plt.savefig("plot_imgs/seq_mean_rel.png")
# plt.clf()