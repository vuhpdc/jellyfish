import argparse
import json
import os


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Convert JSON to TXT file")

    parser.add_argument("--input_file", type=str,
                        help='Input JSON file')

    parser.add_argument("--output_dir", type=str, default="output_detection",
                        help='output JSON file')

    parser.add_argument("--gt", action='store_true', help='Is ground truth')
    parser.set_defaults(gt=False)

    args = parser.parse_args()
    args_dict = args.__dict__
    print("------------------------------------")
    print("Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    return args


def convert(opts):
    input_file = opts.input_file
    output_dir = opts.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = {}
    # read lines from the input json file
    with open(input_file) as json_data:
        input_lines = json.load(json_data)
        for input_line in input_lines:
            image_id = input_line['image_id']
            if image_id not in output:
                output[image_id] = []
            output_line = [input_line['category_id'],
                           input_line['bbox'], input_line['score']]
            output[image_id].append(output_line)

    # write lines to the output txt file
    for image_id in output:
        filename = output_dir + "/" + "{:05d}.txt".format(image_id)
        with open(filename, "w") as file:
            for line in output[image_id]:
                class_id = line[0]
                left = line[1][0]
                right = line[1][1]
                width = line[1][2]
                height = line[1][3]

                if opts.gt:
                    score = line[2]
                    output_line = "{0} {1} {2} {3} {4} {5}\n".format(
                        class_id, score, left, right, width, height)
                else:
                    score = line[2]
                    output_line = "{0} {1} {2} {3} {4} {5}\n".format(
                        class_id, score, left, right, width, height)
                file.write(output_line)


if __name__ == "__main__":
    args = parse_opts()

    convert(args)
