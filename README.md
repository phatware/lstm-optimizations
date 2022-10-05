# New Activation function for ML

## Alternative to tanh() activation function

I present a new activation function that may improve performance and
reduce training time of LSTM RNN or similar networks that contain large
number of neurons using tanh() activation function. This repo contains
research materials on this new activation function.

![](images/image1.png){width="1.4166666666666667in"
height="1.0625in"}

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

![](images/image2.png){width="6.502777777777778in"
height="4.314583333333333in"}

### Derivative

![](images/image3.png){width="1.9166666666666667in"
height="1.0625in"} or
![](images/image4.png){width="1.7291666666666667in"
height="1.1145833333333333in"}

![](images/image5.png){width="6.496527777777778in"
height="4.313888888888889in"}

### Integral

![](images/image6.png){width="1.375in"
height="0.8854166666666666in"}

Performance
![](images/image7.png){width="6.501388888888889in"
height="4.15625in"}

Performance Analysis

See notebooks in the repo. For example, below is the chart of a small
LSTM NN training and prediction performance improvement when using the
new activation function instead of tanh(). Network size between 10 and
300 LSTM neurons doing multivariant weather prediction.

![](images/image8.png){width="6.495833333333334in"
height="2.5256944444444445in"}

Note that the accuracy with the same hyperparameters is practically not
affected.

![](images/image9.png){width="6.495138888888889in"
height="2.3868055555555556in"}

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
