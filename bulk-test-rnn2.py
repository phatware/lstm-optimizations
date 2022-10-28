
import os
import argparse
import subprocess

# create parser
parser = argparse.ArgumentParser()

PATH = "."

# Boolean flag (does not accept input data), with default value
parser.add_argument('-mt', action="store_true", default=False)

# Cast input to integer, with a default value
parser.add_argument('-it', type=int, default=1100)

parser.add_argument('-path', type=str, default=PATH)

args = parser.parse_args()

print(args.mt)
print(args.it)
print(args.path)

neurons_list = [16, 32, 64, 128, 200, 250]

env_copy = os.environ.copy()
env_copy["PATH"] = os.pathsep.join(["."])

for layers in range(2,7):
    for neurons in neurons_list:
        filename = "progress_"
        cmd = [args.path + "/3rd-party/recurrent-neural-net/net", args.path + "/3rd-party/bbc/all.txt", "-it", str(args.it), "-L", str(layers), "-N", str(neurons)]
        if args.mt:
            cmd.append("-mt")
            filename += "mt.csv"
        else:
            filename += "st.csv"
        cmd.append("-pn")
        cmd.append(args.path + "/" + filename)
        cmd.append("-pf")
        cmd.append("100")
        cmd.append("-ap")

        subprocess.call(cmd)
