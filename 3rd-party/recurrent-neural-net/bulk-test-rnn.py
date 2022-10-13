
import os
import argparse
import subprocess

# create parser
parser = argparse.ArgumentParser()

PATH = "/home/stan/work/ML/tanf"

# Boolean flag (does not accept input data), with default value
parser.add_argument('-tf', action="store_true", default=False)

# Cast input to integer, with a default value
parser.add_argument('-it', type=int, default=50000)

parser.add_argument('-path', type=str, default=PATH)

args = parser.parse_args()

print(args.tf)
print(args.it)
print(args.path)

neurons_list = [32, 64, 128, 256, 400, 512]

env_copy = os.environ.copy()
env_copy["PATH"] = os.pathsep.join(["."])

for layers in range(1,4):
    for neurons in neurons_list:
        filename = "progress_" + str(layers) + "x" + str(neurons)
        cmd = [args.path + "/3rd-party/recurrent-neural-net/net", args.path + "/3rd-party/bbc/all.txt", "-it", str(args.it), "-L", str(layers), "-N", str(neurons)]
        if args.tf:
            cmd.append("-tf")
            filename += "_tanf.csv"
        else:
            filename += "_tanh.csv"
        cmd.append("-pn")
        cmd.append(args.path + "/" + filename)

        subprocess.call(cmd)
