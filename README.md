# New Activation function for ML

## Alternative to tanh() activation function

I present a new activation function that may improve performance and
reduce training time of LSTM RNN or similar networks that contain large
number of neurons using tanh() activation function. This repo contains
research materials on this new activation function.

![Activation Function](images/media/image1.png)

## Motivation

-   Tanh() is the slowest (to compute) activation function used in ML.
    In LSTM neurons there are 2 independent tanh()

-   Depending on the CPU/C++ compiler implementation (usually high
    performance ML code is written in C++) tanh() maybe relatively slow,
    especially on thin clients and IoT devices

-   Reducing NN training time saves money!

-   Providing faster classification or prediction results maybe
    important in mission critical applications

-   ...

## Function attributes

Here I compare the above activation function with tanh() and softplus()
activation functions used in many ML packages such as Tensorflow, Torch
and others.

Below: tanh() -- BLUE, f() -- RED, softplus() -- GREEN.

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

## TODO

-   Name this function!

-   Natively (C++!) Add new activation function to a commonly used ML
    package such as Torch or Tensorflow and replace tanh() activation
    with it in LSTM neuron by default (others?)

    -   Performance tests must be done natively

-   Test the new activation function with various networks. Try
    different network sizes and hyperparameters

    -   Research only networks that contain large number of tanh()
        activations

-   Test accuracy when using new activation function and compare results
    to same networks using tanh()

    -   This maybe done in python or other scripting language

-   Test on CPU and GPU

-   ...
