# Jellyfish
Timely Inference Serving for Dynamic Edge Networks, published at IEEE RTSS 2022.

# Requirements
## System
We need three separate Linux (e.g., Ubuntu 18.04.6 LTS) machines for the following three roles respectively.
1. **Server** equipped with one or more GPUs (powered with tensor cores, e.g., NVIDIA RTX2080Ti) to run the DNN serving system.
2. **Client** to emulate multiple (more than one) user devices.
3. **Experiment Manager** to coordinate with the Server and Client to run multiple experiments, save results, plot figures, etc.

## Software
1. Enable password-less SSH communication between these three machines.
2. Install python version ~ 3.6.x (e.g., 3.6.9) 
3. Pytorch >= 1.9.0 compatible with your installed CUDA version (e.g., 10.2 or 11.1).
4. Install [tc-tbf](https://man7.org/linux/man-pages/man8/tc-tbf.8.html) for network bandwidth shaping on the Client machine.
5. IBM CPLEX Optimization Studio (optional) if you want to try out simulation scripts.

# Getting Started

Clone the Jellyfish repository in the HOME directory on *ALL* three machines.
```bash
$ git clone https://github.com/vuhpdc/jellyfish.git
$ cd ${HOME}/jellyfish
```

Install required python modules on *ALL* three machines. This can take a couple of minutes to finish.
```bash
$ pip3 install -r requirements.txt
```

Download pre-trained YOLOv4 PyTorch DNNs, datasets, and ground truths by executing the following command on *ALL* three machines.
```bash
$ ./download_data.sh
```

Generate protobuf source files on the Server and Client machines.
```bash
$ ./src/protos/generate_proto.sh
```

# Source directory structure
```misc
$ ~/jellyfish                   → root directory
	src/ 
		server/             	→ source files for the server component
		client/             	→ source files for client devices
		experiment_manager/ 	→ source files for running experiments
        	simulation/             → source files for running simulation scripts
	datasets/               	→ contains MS-COCO dataset and three traffic videos
	pytorch_yolov4/ 
		models/             	→ contains YOLOv4 pre-trained DNNs
		profiles/           	→ accuracy and latency profiles of the DNNs
    	plots/                  	→ scripts to plot figures after experiments
	logs/                   	→ contains all logs and stats files.
	network_shaping/        	→ contains network traces and tc shaping script
```

# Experiments
## A. DNN Profiling
Before we start any experiment, we should profile DNNs for their accuracy and execution latency on GPUs with different batch sizes.

1. **Accuracy:** For accuracy profiling, we used the [pytorch-YOLO4](https://github.com/Tianxiaomo/pytorch-YOLOv4) repo to create multiple DNN models. We have provided the pre-trained DNNs and their accuracy on the COCO dataset in the download step. The accuracy values are in the file `pytorch_yolov4/profiles/accuracy.txt`

2. **Latency:** To measure the latencies of each DNN, run the following command on the Server. It may take around 5-10 hours. 
    ```bash
    server-$ cd src/server/profiler/
    server-$ ./run.sh  
    ```
    This command will run with default parameters such as 2 GPUs, 16 DNNs, a maximum batch size of 12, and so on. The custom values can be added in the [run.sh](./src/server/profiler/run.sh) script or [opts.py](./src/server/profiler/opts.py) file.

    If you have only one GPU installed on the server, you can use the following command instead.

    ```bash
    server-$ num_gpus=1 ./run.sh
    ```

**Output:** The profiles are saved in the directory `pytorch_yolov4/profiles/`.

**Note:** Once you start the DNNs profiling, the previously downloaded profiles will be overwritten. To fetch our profiles again, you have to run the [download_data.sh](./download_data.sh) script again.

## B. Start Server
Once the DNN profiles are generated, we need to start the multi-process server which runs the GRPC endpoint, DNN workers and executors, scheduler, and dispatcher. 
```bash
server-$ cd src/server/
server-$ ./run.sh
```
It starts the Server with default parameters such as 2 GPUs, 16 DNNs, a periodic scheduling interval of 0.5s for scheduler, 5 active DNN models cached on one GPU, and so on. These parameters can be parameterized in the [run.sh](./src/server/run.sh) script or [opts.py](./src/server/opts.py) file. 

If you have only one GPU installed on the Server, you can use the following command.
```bash
server-$ num_gpus=1 ./run.sh
```
**Output:** The logs and stats are saved locally at `logs/server/`.

## C. Start Experiments
Once the server has started, we can start experiments from the Experiment Manager machine. You may want to start experiments in the `screen` or `tmux` session because the experiments may take anywhere between 10-30 hours, depending on the parameters.
```bash
experiment_manager-$ cd src/experiment_manager
```

The Experiment Manager sends commands to the Server and Clients to start video streaming and inference serving. We can start the experiments with default parameters. For example, number of clients = [1, 2, 4, 8], clients SLO = [75, 100, 150], clients FPS = [15, 25], synthetic network trace, and so on. One experiment setting is run for all three traffic videos and three iterations.
```bash
experiment_manager-$ server_ip=<server_ip> \
                    server_username=<server_username> \
                    server_ssh_port=22 \
                    client_ip=<client_ip> \
                    client_username=<client_username> \
                    client_ssh_port=22 \
                    total_iter=1 \
                    network_trace_type=synthetic_trace \
                    ./run_experiments.sh
```

You should provide the details of the Server (such as IP, username), Client (such as IP, username, network interface), network trace type (`synthetic_trace`, `wifi_trace`, `lte_upload_trace`), and so on in the command line. Alternatively, you can add it in the [run_experiments.sh](./src/experiment_manager/run_experiments.sh) script.
 
By default, the experiment manager assumes all the clients should be run on the same Client machine. To use more than one Client machine, you may have to provide details in the `clients_cfg/client_cfg_template_*` files. Check the template files in the directory [clients_cfg](./src/experiment_manager/clients_cfg) on how to specify client configurations.

	
**Output:** All the logs and stats along with clients accuracy will be saved locally at `logs/experiment_manager/<network_trace_type>`.

**Note:**
You can see whether the experiments have started correctly or not by checking the stats/logs on the Client machine.
```bash
client-$ tail -f logs/client/host_0/frame_stats.csv
```

If you want to *abruptly*  stop the experiment in the middle, you have to first stop the Client processes and then stop the process on the Experiment Manager.
```bash
client-$ pkill run.sh; pkill python3; sudo pkill shape_tbf
```

## D. Plot Figures

Once the experiment has been finished, we can plot the figures reported in the paper.
	
1. DNN profiles (**Fig. 6**)
    ```bash
    experiment_manager-$ cd plots/fig_6
    experiment_manager-$ ./run.sh
    ```

2. DNN adaptation (**Fig. 7**)
    ```bash
    experiment_manager-$ cd plots/fig_7
    experiment_manager-$ ./run.sh
    ```

Similarly, you can plot **Fig. 8** (accuracy vs miss rate) and **Fig. 9** (latency CDF). If you have run the experiments for only one iteration then you have to set the env variable `total_iter=1`.

```bash
experiment_manager-$ total_iter=1 ./run.sh
```

**Output:** All the figures will be saved in their respective directory as a `fig_*.pdf`.

Finally, you can stop the Server using the following command.
```bash
server-$ pkill --signal 15 run.sh
```

## E. Simulation
You can also play with the simulation scripts. Please check the run scripts in the directory [src/simulation](./src/simulation). 
```bash
$ cd src/simulation
$ ./run_compare.sh  # compare accuracy of the SA-DP algorithm
$ ./run_timings.sh  # Check timings of the SA-DP algorithm
```

# Note
This repository is a cleaned-up version of the code used for Jellyfish experiments. If you find any bug or issue in the code (or in the paper), please do let us know. 

Moreover, if you find this code or paper useful, then please do cite our work.

```bib
@inproceedings{nigade2022jellyfish,
  title={{Jellyfish}: Timely Inference Serving for Dynamic Edge Networks,
  author={Vinod Nigade and Pablo Bauszat and Henri Bal and Lin Wang},
  booktitle={{IEEE} Real-Time Systems Symposium ({RTSS})},
  year={2022}
}
```

We are thankful to the authors of [pytorch-YOLO4](https://github.com/Tianxiaomo/pytorch-YOLOv4), [dds](https://github.com/KuntaiDu/dds/) and [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)  for making their code/data public.
