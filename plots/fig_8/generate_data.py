import pandas as pd
import numpy as np
import ast
import argparse

# Global variables
CLIENTS_TESTED = [1, 2, 4, 8]
CLIENTS_SLO = [75, 100, 150]
CLIENTS_FPS = [15, 25]
VIDEOS = ["dds/trafficcam_1", "dds/trafficcam_2", "dds/trafficcam_3"]
CLIENT_HOST_IDS = []
for i in range(16):
    CLIENT_HOST_IDS.append(f"host_{i}")


def deadline_misses(df, target_lat):
    total = len(df)
    df = df[df["dropped_frame"] == 0]
    count = len(df[df["e2e_latency"] > target_lat])
    dropped_frames = total - len(df)
    print("Dropped frames: {:.2f}".format((dropped_frames / total)*100))
    count += dropped_frames
    miss_rates = (count / total) * 100
    print(f"Total Misses: {miss_rates}")
    return miss_rates, count, total


def compute_misses_for_the_setup(args, experiment_type, clients_slo_lst, clients_fps_lst, total_clients, iter):
    clients_df = {}
    print(f"\nMisses: {experiment_type}")
    for video_name in VIDEOS:
        clients_df[video_name] = []
        for i in range(total_clients):
            df = pd.read_csv(
                f"{args.data_dir}/{experiment_type}/{video_name}/iter_{iter}/{CLIENT_HOST_IDS[i]}/frame_stats.csv", delimiter=",")
            df.loc[(df.dropped_frame == 1), 'used_model'] = None
            clients_df[video_name].append(df)

    total_count = 0
    total_misses = 0
    for video_name in VIDEOS:
        print(f"Video: {video_name}")
        for i, df in enumerate(clients_df[video_name]):
            print(f"\tClient {i}")
            miss_rate, _miss_count, _total_count = deadline_misses(
                df, target_lat=clients_slo_lst[i])
            print(f"\t\tMisses: {_miss_count}, Total: {_total_count}")
            total_misses += _miss_count
            total_count += _total_count

    total_miss_rate = (total_misses/total_count)*100
    print(f"Aggregate Misses: {total_misses}/{total_count}", total_miss_rate)
    return total_miss_rate


def compute_misses_for_each_slo_and_fps(args, clients_slo, clients_fps, total_iter=1):
    miss_rates_dict = {}
    # Print deadline misses
    for total_clients in CLIENTS_TESTED:
        miss_rates = []
        for iter in range(total_iter):
            print(f"\tIter: {iter}")
            clients_slo_lst = [clients_slo] * total_clients
            clients_fps_lst = [clients_fps] * total_clients
            experiment_type = f"client_cfg_template_{total_clients}_{clients_fps}_{clients_slo}"
            deadline_misses = compute_misses_for_the_setup(args,
                                                           experiment_type,
                                                           clients_slo_lst,
                                                           clients_fps_lst,
                                                           total_clients,
                                                           iter)
            miss_rates.append(deadline_misses)

        miss_rates_dict[total_clients] = np.array(miss_rates)
    return miss_rates_dict


# Print accuracy
def compute_f1_for_the_setup(args, clients_slo, clients_fps, total_clients, iter):
    experiment_type = f"client_cfg_template_{total_clients}_{clients_fps}_{clients_slo}"
    print(f"\nAccuracy F1: {experiment_type}")
    TP, FP, FN = 0, 0, 0
    for video_name in VIDEOS:
        print(f"Video: {video_name}")
        for i in range(total_clients):
            print(experiment_type,
                  CLIENT_HOST_IDS[i], f"{args.data_dir}/{experiment_type}/{video_name}/iter_{iter}/{CLIENT_HOST_IDS[i]}")
            with open(f"{args.data_dir}/{experiment_type}/{video_name}/iter_{iter}/{CLIENT_HOST_IDS[i]}//accuracy_F1.txt") as acc_file:
                acc_dict = ast.literal_eval(acc_file.read())

            all_frames_f1_dict = acc_dict["Detected Frames F1"]
            print(all_frames_f1_dict)
            print("\tClient {}: Precision:{}, Recall: {}, F1: {}".format(
                i, all_frames_f1_dict['precision'], all_frames_f1_dict['recall'], all_frames_f1_dict['F1']))
            TP += all_frames_f1_dict['total TP']
            FP += all_frames_f1_dict['total FP']
            FN += all_frames_f1_dict['total FN']

    precision = round(TP/(TP+FP), 5)
    recall = round(TP/(TP+FN), 5)
    f1 = round((2.0*TP/(2.0*TP+FP+FN)), 5)
    print(
        f"Total: TP = {TP}, FP = {FP}, FN = {FN}, Precision:{precision}, Recall: {recall}, F1: {f1}")
    return f1


def compute_f1_for_each_slo_and_fps(args, clients_slo, clients_fps, total_iter=1):
    f1_scores_dict = {}
    for total_clients in CLIENTS_TESTED:
        f1_scores_for_each_iter = []
        for iter in range(total_iter):
            f1 = compute_f1_for_the_setup(args,
                                          clients_slo,
                                          clients_fps,
                                          total_clients,
                                          iter)
            f1_scores_for_each_iter.append(f1)
        f1_scores_dict[total_clients] = np.array(f1_scores_for_each_iter)
    return f1_scores_dict


def generate_data(args):
    print_str = "Clients,SLO,FPS,F1,MissRate,MissRate_std\n"
    for clients_slo in CLIENTS_SLO:
        for clients_fps in CLIENTS_FPS:
            miss_rates_dict = compute_misses_for_each_slo_and_fps(args,
                                                                  clients_slo,
                                                                  clients_fps,
                                                                  total_iter=args.num_iter)
            f1_scores_dict = compute_f1_for_each_slo_and_fps(args,
                                                             clients_slo,
                                                             clients_fps,
                                                             total_iter=args.num_iter)

            for total_clients in CLIENTS_TESTED:
                miss_rates = miss_rates_dict[total_clients]
                f1_scores = f1_scores_dict[total_clients]

                print(
                    f"Clients {total_clients}: MissRate: {round(miss_rates.mean(), 5)} ({round(miss_rates.std(), 5)}), F1Score: {round(f1_scores.mean(), 5)} ({round(f1_scores.std(), 5)}")
                print_str += f"{total_clients},{clients_slo},{clients_fps},{round(f1_scores.mean(), 5)},{round(miss_rates.mean(), 5)},{round(miss_rates.std(), 5)}"
                print_str += "\n"

    # Save this to the file
    with open(f'{args.output_dir}/data.csv', "w") as f:
        f.write(print_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory containing log files")
    parser.add_argument("--output_dir",
                        help="Directory to save the extracted information")
    parser.add_argument("--num_iter",
                        help="Number of iterations to be considered",
                        type=int)

    args = parser.parse_args()
    generate_data(args=args)
