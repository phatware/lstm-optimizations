# Recurrent Neural Network: Evaluating the Inverse Square Linear Unit Function

In the ongoing quest for optimizing neural network performance, the testing of activation functions plays a pivotal role. Central to this exploration is the comparison of the Inverse square linear unit function, given by \(x/\sqrt{1+x^2}\), and the traditional `tanh(x)` function. Using a Recurrent Neural Network (RNN) as a foundational structure offers a compelling landscape for this evaluation. 

When constructing systems designed to adapt and recognize patterns, researchers often delve into foundational theories regarding the operation of intricate biological systems, such as human brains. One of the captivating discoveries in this realm is the RNN. An RNN is a sophisticated system with feedback loops, possessing the capability to retain past information. Such a system is indispensable for modeling long-term dependencies, especially those found in natural language processing.

## Build 

* On Mac or Linux:
```Bash
# Build using cmake
make
```

* On Windows, use Microsoft Visual Studio 

## Running

Then run the program:
```Bash
./net datafile 
```

where datafile is a file with the training data and it will start training on it. You can see the progress 
over time. 

## Configuration

To fine-tune the program's behavior, refer to the "std_conf.h" file. This file facilitates the adjustment of hyperparameters, layer settings, output frequency, and more. Moreover, it provides an option to utilize either the Inverse square linear unit function or `tanh(x)` for performance testing. Modifying this file necessitates recompilation using the `make` command. Additionally, certain behaviors can be adjusted using input arguments.

Running the program with no arguments triggers the help output to be displayed. This help shows what flags can be
passed as arguments to the program to modify its behavior. The output looks like this:

<pre>
Usage: ./net datafile [flag value]*

Flags can be used to change the training procedure.
The flags require a value to be passed as the following argument.
    E.g., this is how you train with a learning rate set to 0.03:
        ./net datafile -lr 0.03

The following flags are available:
    -r  : read a previously trained network, the name of which is currently configured to be 'lstm_net.net'.
    -lr : learning rate that is to be used during training, see the example above.
    -it : the number of iterations used for training (not to be confused with epochs).
    -ep : the number of epochs used for training (not to be confused with iterations).
    -mb : mini batch size.
    -dl : decrease the learning rate over time, according to lr(n+1) <- lr(n) / (1 + n/value).
    -st : number of iterations between how the network is stored during training. If 0 only stored once after training.
    -out: number of characters to output directly, note: a network and a datafile must be provided.
    -L  : Number of layers, may not exceed 10
    -N  : Number of neurons in every layer, may not exceed 300
    -vr : Verbosity level. Set to zero and only the loss function after and not during training will be printed.
    -c  : Don't train, only generate output. Seed given by the value. If -r is used, datafile is not considered.
    -s  : Save folder, where models are stored (binary and JSON).
    -pf : Store progress at specified iteration intervals. Default is 1000
    -tf : Use isru() instead of tanh().
    -mt : Use separate threads for each layer.
    -pn : Progress file name. default is progress.csv
    -ap : Append progress file instead of overwriting.

Check std_conf.h to see what default values are used, these are set during compilation.

./net (double precision) compiled on Nov  4 2022 at 12:50:43 with Ubuntu Clang 14.0.0
</pre>

Notably, the -st flag is particularly useful. By default, the network saves upon program interruption using Ctrl-C. However, with this argument, the network saves periodically during training, safeguarding against unexpected program termination.

## Experiment Outcomes

The program was trained on the initial Harry Potter book, producing intriguing outputs such as:

```vbnet
Iteration: 303400, Loss: 0.07877, output: ed Aunt Petunia suggested
timidly, hours later, but Uncle Vernon was pointing at what had a thing while the next day. The Dursleys swut them on, the boy.
```
Although the text reflects a rudimentary understanding of English language structures, it is noteworthy that the program was only trained at the character code level. For further endeavors based on this neural network, refer to the corresponding Twitter bot.

## Mathematical Overview

The model utilizes a specific design for a single LSTM cell. An illustrative image delineates a single-layer forward pass. Codebase terminology stems from this notation. A key feature of this implementation is the option to test the performance of the Inverse square linear unit function against `tanh(x)`. For clarity, terms like `model->Wf` pertain to the 'Wf' depicted in the aforementioned set of expressions, and the `-tf` flag allows users to switch between the two activation functions during the training process. Layers are interlinked via a fully connected layer, incorporating 'Wy' and 'by'. Inputs to this connected layer are the 'h' array from the preceding expressions. Finally, a softmax application is made to the model's output layer to derive probabilities.
