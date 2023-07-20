#include "matrix.h"
#include <string.h>
#include "interval.h"
#include "lp_dev/lp_lib.h"
#include <time.h>

#ifndef NNET_H
#define NNET_H

/* outward rounding */
#define NEED_OUTWARD_ROUND 0
#define OUTWARD_ROUND 0.00000005

/* which property to test */
extern int PROPERTY;

/* log file */
extern char *LOG_FILE;
extern FILE *fp;

typedef int bool;
enum { false, true };

/*
 * Network instance modified from Reluplex
 * malloc all the memory needed for network
 */
struct NNet 
{
    int symmetric;     
    int numLayers;     
    int inputSize;     
    int outputSize;    
    int maxLayerSize;  
    int *layerSizes;   

    float *mins;      
    float *maxes;     
    float *means; 
    float *ranges;    
    float ****matrix;
                       
    struct Matrix* weights;
    struct Matrix* bias;

    int target;
    int *feature_range;
    int feature_range_length;
    int split_feature;
};


/* load the network from file */
struct NNet *load_network(const char *filename, int target);

/* free all the memory for the network */
void destroy_network(struct NNet *network);


/* load the input range of the property */
void load_inputs(int PROPERTY, int inputSize, float *u, float *l);


/* denormalize input */
void denormalize_input(struct NNet *nnet, struct Matrix *input);


/* denormalize input range */
void denormalize_input_interval(struct NNet *nnet, struct Interval *input);


/* normalize input */
void normalize_input(struct NNet *nnet, struct Matrix *input);


/* normalize input range */
void normalize_input_interval(struct NNet *nnet, struct Interval *input);


/*  
 * Uses sgemm to calculate the output
 * 0.00001359 sec for one run with one core
*/
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output);


// new propagation method which can make the verification much faster! 
int forward_prop_interval_equation_linear2(struct NNet *network, struct Interval *input, struct Interval *output, struct Interval *grad);

/*
 * The back prop to calculate the gradient
 * 0.000249 sec for one run with one core
*/
void backward_prop(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize]);


#endif
