echo building...
rm *.txt
gcc -O3 -std=gnu++20 -lstdc++ -lm -Wall  lstm-basic-test/DataProcessor.cpp lstm-basic-test/FileProcessor.cpp lstm-basic-test/LSTMNet.cpp lstm-basic-test/lstm-basic-test.cpp
echo running...
./a.out ./data
rm a.out
