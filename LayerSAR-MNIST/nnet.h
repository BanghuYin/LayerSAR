#include "matrix.h"
#include <string.h>
#include "interval.h"
#include "lp_dev/lp_lib.h"
#include <time.h>
#include <math.h>

#ifndef NNET_H
#define NNET_H

typedef int bool;
enum { false, true };

#define NEED_OUTWARD_ROUND 0
#define OUTWARD_ROUND 0.00000005
#define MAX_PIXEL 255.0
#define MIN_PIXEL 0.0
#define MAX 1
#define MIN 0



extern int PROPERTY;
extern float INF;
extern float Timeout;
extern struct timeval start,finish, last_finish;

//Neural Network Struct
struct NNet 
{
    int symmetric;     //1 if network is symmetric, 0 otherwise
    int numLayers;     //Number of layers in the network
    int inputSize;     //Number of inputs to the network
    int outputSize;    //Number of outputs to the network
    int maxLayerSize;  //Maximum size dimension of a layer in the network
    int *layerSizes;   //Array of the dimensions of the layers in the network
    int *layerTypes;   //Intermediate layer types

    int convLayersNum;
    int **convLayer;
    float ****conv_matrix;
    float **conv_bias;

    float min;      //Minimum value of inputs
    float max;     //Maximum value of inputs
    float mean;     //Array of the means used to scale the inputs and outputs
    float range;    //Array of the ranges used to scale the inputs and outputs
    float ****matrix; //4D jagged array that stores the weights and biases
                       //the neural network.
    struct Matrix* weights;
    struct Matrix* bias;
    struct Matrix* pos_weights;
    struct Matrix* neg_weights;

    int target;
    int *feature_range;
    int feature_range_length;
    int split_feature;
};

struct NNet *load_conv_network(const char *filename, int img);
void destroy_conv_network(struct NNet *network);

void load_inputs(int img, int inputSize, float *input);

void initialize_input_interval(struct NNet *nnet, int img, int inputSize, float *input, float *u, float *l);

int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

int forward_prop_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);

float set_output_constraints(lprec *lp, float *equation, int start, int *rule_num, int inputSize, int is_max, float *output, float *input_prev);

void denormalize_input(struct NNet *nnet, struct Matrix *input);

void normalize_input(struct NNet *nnet, struct Matrix *input);

void normalize_input_interval(struct NNet *nnet, struct Interval *input);

void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num);

#endif
