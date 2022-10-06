# New Activation function for ML

## A faster alternative to tanh() activation function

I present a new activation function that may improve NN performance and
reduce training time of LSTM RNN or similar networks that contain large
number of neurons using tanh() activation function. This repo contains
research materials on this new activation function.

![Activation Function](images/media/image1.png)

## Motivation

-   Tanh() is the slowest (to compute) activation function used in ML.
    For example, in LSTM neurons there are 2 independent tanh()

-   Depending on the CPU/C++ compiler implementation (usually high
    performance ML code is written in C++) tanh() maybe relatively slow,
    especially on thin clients and IoT devices

-   Reducing NN training time saves money! And energy! 
    *This function is environmentally friendly!*

-   Providing even marginally faster classification or prediction results maybe
    important in mission critical applications

-   Are there any other function's characteristics that may be beneficial
    in ML applications?

-   Preliminary results show 1-2% acceleration of training, classification 
    and prediction with LSTM networks of relatively small size, while no notable 
    decrease in learning rate or accuracy was detected.

-   Expect even better results with larger LSTM RNNs

## Function attributes

Here I compare the above activation function with tanh() and softplus()
activation functions used in many ML packages such as Tensorflow, Torch
and others.

Below: **tanh() -- BLUE, f() -- RED, softplus() -- GREEN**.

### Function
![Plot f(x), tanh(x), softplus(X)](images/media/image2.png)

### Derivative
![Derivative](images/media/image3.png) or ![](images/media/image4.png)

![Derivative Plot](images/media/image5.png)

### Integral
![](images/media/image6.png)

![Integral Plot](images/media/image7.png)

## Performance Analysis

See notebooks in the repo. For example, below is the chart of a small
LSTM NN training and prediction performance improvement when using the
new activation function instead of tanh(). Network size between 10 and
300 LSTM neurons doing multivariant weather prediction.

![Measurable performance improvement](images/media/image8.png)

Note that the accuracy with the same hyperparameters is practically not
affected.

![Same accuracy as tanh()](images/media/image9.png)

## What's in this repo

This repo contains research resources and test apps. Try running test apps on 
your computer and record test results.

### math-bench

Simple app to measure performance of different math functions. 

#### Linux/WSL and MAC OS

To build and run the test execute `math-bench/test.sh`

* Xcode command-line tools is required to compile this app.

#### Windows 

Open `math-bench\math-bench.sln` file with Visual Studio 2022. Run the solution.

* Microsoft Visual Studio 2022 Community edition or better is required.

### lstm-basic-test

Very simple native (C++) LSTM network performance test. This is Up to 300 LSTM neurons, 
single layer multivariate weather predictor network.

#### Linux/WSL and MAC OS

To build and run the test execute `lstm-basic-test/test.sh`

* Xcode command-line tools is required to compile this app.

#### Windows 

Open `lstm-basic-test\lstm-basic-test.sln` file with Visual Studio 2022. Run the solution.

* Microsoft Visual Studio 2022 Community edition or better is required.

### Visualize Results

Use provided jupyter notebooks to visualize results.

* Python 3.8 with Tensorflow 2.7 or later is required 
* To visualize `math-bench` results, open `performance_charts.ipynb`
* To visualize `lstm-basic-test` results, open `lstm-stats.ipynb`

## Next Steps / TODO

-   Name this function!

-   Natively (in C++) add new activation function to a commonly used ML
    package such as Torch or Tensorflow then replace tanh() activation
    in LSTM neuron by default (others?)

    -   Performance tests must be done natively

-   Test the new activation function with various networks. Try
    different network sizes and hyperparameters

    -   Research only networks that contain large number of tanh()
        activations

-   Test accuracy when using new activation function comparing to tanh()

    -   Assuming computational performance testing here is not a goal,
        this can done in python or other scripting language (R?)

-   Test on CPU and GPU (Cuda)

    -   I included a very basic benchmark test compiled with CUDA libraries on 
        Windows, but I am not sure GPU actually computes advanced math functions
        like exp() or tanh()
    
    -   More research on GPU is needed

-   Start and keep updating list of references for future publications

-   Write/publish a research paper, when confident with test results and have enough 
    research materials/references

    -   *Is it patentable?* I don't think the function itself is, but maybe there is 
        a specific application of this or similar equation besides a generic activation
        (like a significant hardware acceleration or new more efficient network type)
        that may be patentable. This needs to be researched.

-   Besides computational performance, explore any other potential benefits 
    of this function

-   Besides LSTM networks, explore other types of NN where tanh() is heavily 
    used

    -   Maybe new types of networks/neurons using this activation function?

-   Was this function researched already for ML? I did not find any references so
    far...

-   Because this function is simpler to compute comparing to tanh() or exp(), is any
    hardware acceleration/GPU possible? Especially for wide classifiers, for example, 
    as an optimization of the dot() product?
    
        -   Note: this function uses only multiplication, division, and addition 
            plus a square root: no loops/repeated operations and square root is fast! 

        -   Possible research area: in some cases it may be possible to avoid square root by 
            squaring the entire NN and only taking square root at the end.


- **TBD**
