import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots.config as config
import argparse
import os
config.configure(mpl, plt)
FPS = [15, 15]

colors = ['tab:blue',  'tab:red', 'tab:orange', 'tab:green']
styles = ['solid', 'dashed', (0, (5, 5)), (0, (3, 5, 1, 5))]

FRAME_SIZES = [128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]

TOTAL_MODELS = 17


def read_profiled_accuracies(profiled_dir, total_models):
    file = os.path.join(profiled_dir, 'accuracy.txt')
    df = pd.read_csv(file, delimiter=",", header=0)
    accuracies = []
    for model_number in range(total_models):
        acc = df.loc[(df["ModelSize"] == FRAME_SIZES[model_number]),
                     "Accuracy"].values
        accuracies.append(acc[0])
    return np.array(accuracies)


def latency_estimator(latencies):
    '''
    Adjust latency measurements. We adjust it because of the measurement noise. We realized that at small batch sizes i.e., 1 or 2,
    Some models have lower latency, within (1-2ms) than the smaller model. We currently attribute this variation to the 
    measurement noise. Generally, our scheduling method assumes that the bigger model have larger latency and ideally it should be
    true. If you do not intend to adjust the curves, then you can adjust the percentile factor to calculate latency.

    You can also use the quantile regression to estimate the P99 latency, as specified in [Clipper, NSDI'17]
    '''
    for model_idx in range(latencies.shape[0]):
        for batch in range(latencies.shape[1]):
            if batch > 0 and latencies[model_idx][batch] < latencies[model_idx][batch-1]:
                correction = latencies[model_idx][batch -
                                                  1] - latencies[model_idx][batch]
                # assert correction < 2.00
                latencies[model_idx][batch] = latencies[model_idx][batch-1]

            if model_idx > 0 and latencies[model_idx][batch] < latencies[model_idx-1][batch]:
                correction = latencies[model_idx -
                                       1][batch] - latencies[model_idx][batch]
                # assert correction < 2.00
                latencies[model_idx][batch] = latencies[model_idx-1][batch]

    return latencies


def read_profiled_latencies(profiled_dir, total_models, max_batch_size, percentile=0.99):
    _SLOWDOWN_FACTOR = 1.0
    latencies = np.zeros((total_models, max_batch_size), dtype=float)
    for model_number in range(total_models):
        file = os.path.join(profiled_dir, 'latency/profile_latency_{}.txt'.format(
            FRAME_SIZES[model_number]))
        df = pd.read_csv(file, delimiter=",", header=0)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        for batch in range(1, max_batch_size+1):
            df_batch = df[df["Batch"] == batch]
            samples = df_batch["InferenceTime"].iloc[0:]
            latencies[model_number, batch -
                      1] = (samples).quantile(percentile) * _SLOWDOWN_FACTOR

    latencies = latency_estimator(latencies)
    return latencies


def compute_model_throughput(latency_mat):
    throughput_mat = []
    for model_index, lat_vec in enumerate(latency_mat):
        throughput_vec = []
        for i, lat in enumerate(lat_vec):

            t = ((i+1) / lat) * 1000
            throughput_vec.append(t)
            if i > 0 and throughput_vec[i-1] > throughput_vec[i]:
                pass
        throughput_mat.append(throughput_vec)
    return np.array(throughput_mat)


def read_data(data_dir):
    # Server
    accuracy_profiles = read_profiled_accuracies(data_dir, TOTAL_MODELS)
    latency_profiles_median = read_profiled_latencies(
        data_dir, TOTAL_MODELS, 8, percentile=0.5)
    latency_profiles_P99 = read_profiled_latencies(
        data_dir, TOTAL_MODELS, 8, percentile=0.99)
    latency_profiles_P99 = latency_estimator(latency_profiles_P99)

    throughput_profiles_P99 = compute_model_throughput(latency_profiles_P99)
    return accuracy_profiles, latency_profiles_median, latency_profiles_P99, throughput_profiles_P99


def plot_data(data_dir, output_dir):
    def plot_lines(ax, data, label, title=""):
        x = list(range(1, data.shape[1]+1))
        for i in range(data.shape[0]):
            ax.plot(x, data[i], label="DNN_{}".format(
                FRAME_SIZES[i]), linewidth=1.5)
        # ax.legend()
        ax.set_ylabel(label, fontsize=14)
        ax.set_xlabel("Batch Size", fontsize=14)
        ax.tick_params("both", labelsize=10)

        xticks = list(range(1, data.shape[1]+1, 1))
        xticklabels = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=10)

        ax.set_title(title, fontsize=14)
        ax.grid(linestyle='dotted', linewidth=0.5)
        return

    def plot_accuracy(ax, accuracy_profiles):
        x = list(range(1, accuracy_profiles.shape[0]+1))
        ax.plot(x, accuracy_profiles, marker="+", linewidth=1.5)
        ax.set_ylabel("mAP", fontsize=14)
        ax.set_xlabel("DNN Input Size", fontsize=14)
        ax.tick_params("both", labelsize=10)
        ax.grid(linestyle='dotted', linewidth=0.5)

        xticks = list(range(1, accuracy_profiles.shape[0]+1, 2))
        xticklabels = FRAME_SIZES[:TOTAL_MODELS:2]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=10, rotation=60)
        ax.set_title("Accuracy", fontsize=14)

    accuracy_profiles, latency_profiles_median, latency_profiles_P99, throughput_profiles_P99 = read_data(
        data_dir)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=config.set_size(
        config.width, height_div=3.5, fraction=0.5))

    # Plot accuracy
    plot_accuracy(ax[0], accuracy_profiles)

    # Plot media latency
    plot_lines(ax[1], latency_profiles_P99,
               'P99 Latency (ms)', "Latency")

    # Plot throughput for P99 latency
    plot_lines(ax[2], throughput_profiles_P99,
               "Frames Per Second", "Throughput")

    # Legend outside of the subplot
    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=[0.52, 1.15],
               labelspacing=0.50, handletextpad=0.3, columnspacing=1.0,
               loc='center', fontsize=10, ncol=(len(labels)//3 + 1), frameon=True)

    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f'{output_dir}/fig_6.pdf', format='pdf',
                dpi=config.dpi, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help="directory containing csv files",
                        default="./data/")
    parser.add_argument("--output_dir",
                        help="Directory to save the output figure",
                        default="./")

    args = parser.parse_args()
    data_dir = args.data_dir

    plot_data(data_dir, output_dir=args.output_dir)
