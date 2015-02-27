#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"
//#define NOMINMAX
//#include "imdebug.h"

void Lenet5_convnet();

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

int main(void) {
    Lenet5_convnet();
}

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void Lenet5_convnet(void) {
    // construct LeNet-5 architecture
    typedef network<mse, gradient_descent_levenberg_marquardt> CNN;
    CNN nn;
    convolutional_layer<CNN, tan_h> C1(32, 32, 5, 1, 6);
    average_pooling_layer<CNN, tan_h> S2(28, 28, 6, 2);
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X
    convolutional_layer<CNN, tan_h> C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16));
    average_pooling_layer<CNN, tan_h> S4(10, 10, 16, 2);
    convolutional_layer<CNN, tan_h> C5(5, 5, 5, 16, 120);
    fully_connected_layer<CNN, tan_h> F6(120, 10);

    assert(C1.param_size() == 156 && C1.connection_size() == 122304);
    assert(S2.param_size() == 12 && S2.connection_size() == 5880);
    assert(C3.param_size() == 1516 && C3.connection_size() == 151600);
    assert(S4.param_size() == 32 && S4.connection_size() == 2000);
    assert(C5.param_size() == 48120 && C5.connection_size() == 48120);

    nn.add(&C1);
    nn.add(&S2);
    nn.add(&C3);
    nn.add(&S4);
    nn.add(&C5);
    nn.add(&F6);

    std::cout << "load testset..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> test_labels;
    std::vector<label_t>result_labels;
    std::vector<vec_t> test_images;

    parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("t10k-images.idx3-ubyte", &test_images);

    // load networks
    std::cout << "load models..." << std::endl;
    std::ifstream ifs("LeNet-weights");
    ifs >> C1 >> S2 >> C3 >> S4 >> C5 >> F6;

    std::cout << "testing..." << std::endl;
    // test and show results
    nn.test(test_images, test_labels, result_labels).print_detail(std::cout);
    //save results
	std::ofstream ofs("Result labels");
	for(int i=0; i< result_labels.size();i++)
	{
		ofs<<result_labels.at(i)<<std::endl;
	}
}



