# Recurrent neural network

In the process of designing systems that can adapt and learn patterns we explore on a basic, fundamental, level theories about how complex biological systems, such as human brains, work. I find this very fascinating. A recurrent neural network is a system that contains feedback loops and can store information from the past. 
This is necessary in order to model long-term dependencies such as can be found in natural language processing. 

This program will learn to produce text similar to the one that
it has been training on using a LSTM network implemented in C. The repo is inspired by Andrej Karpathys <i>char-rnn</i>: https://github.com/karpathy/char-rnn but instead implemented in C to be used in more constrained environments.

# Build 

* On Mac or Linux:
```Bash
# Build using cmake
make
```

* On Windows, use Microsoft Visual Studio 

# Running

Then run the program:
```Bash
./net datafile 
```

where datafile is a file with the training data and it will start training on it. You can see the progress 
over time. 

# Configure default behavior before build

Check out the file "std_conf.h".

In std_conf.h you can edit the program. You can edit the hyperparameters such as learning rate etc, set the number of layers (2/3 is best I think), set how often it should output data etc. If you edit this file, you edit the source code and you will need to rebuild the program with the command "make". You can also use input arguments to set some of the behavior.

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
    -N  : Number of neurons in every layer
    -vr : Verbosity level. Set to zero and only the loss function after and not during training will be printed.
    -c  : Don't train, only generate output. Seed given by the value. If -r is used, datafile is not considered.
    -s  : Save folder, where models are stored (binary and JSON).
    -tf : Use tanf() instead of tanh()
    -td : When using tanf(), use simplified derivative in back propagation (drop power of 3/2)

Check std_conf.h to see what default values are used, these are set during compilation.

./net compiled Jan 18 2021 13:08:35
</pre>

The -st flags is great. Per default the network is stored upon interrupting the program with Ctrl-C. But using this argument, you can let the program train and have it store the network continuously during the training process.
In that case the network is available for you even if the program is unexpectedly terminated.

Enjoy! :)

# Examples
I trained this program to read the first Harry Potter book, It produced quotes such as this: 

```
Iteration: 303400, Loss: 0.07877, output: ed Aunt Petunia suggested
timidly, hours later, but Uncle Vernon was pointing at what had a thing while the next day. The Dursleys swut them on, the boy.
```

It has definitely learned something as it is writing in english, and I only model on the most
general basis, the character codes.

For more activity based on this neural network, check out my twitter bot: 
https://twitter.com/RicardicusPi

# Mathematical expressions

This is the model that is used for a single LSTM cell: 

<img src="https://raw.githubusercontent.com/Ricardicus/recurrent-neural-net/master/html/LSTM_forward.png"></img>

That image describes a single layer forward pass. The terminology in the codebase are derived from that 
notation. So for example, if you wonder what model->Wf means, then Wf is represented in that set of 
expressions. Also, model->dldWf means the backpropagated gradients for Wf. I connect the layers with a
fully connected layer, introducing Wy and by also. Inputs to the fully connected is the array h in the 
set of expressions above. I apply a softmax to output layer of the model also in the end,
to get the probabilities.
