#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include<signal.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "nnet.h"

#ifndef SPLIT_H
#define SPLIT_H


/* Print detailed progress */
extern int NEED_PRINT;

extern int SPLITING_METHOD;
extern int RELAX_MOD;//0 for ours, 1 for POPL, 2 for Neurify
extern int FUL_or_RAND;// 0 for FUL , 1 for random
extern int MAX_THREAD;//
extern int REFINE_delay_NUM; // 0 for no refinement, default:1

extern int DEBUG;


/* No bisection mode */
extern int NEED_FOR_ONE_RUN;

/* set 1 if a concrete adversarial example is
found */
extern int adv_found;
extern int forward_count;
extern int m_depth;

/* Mode for faster search concrete
adversarial examples */
extern int CHECK_ADV_MODE;
extern int PARTIAL_MODE;

/* Bisection tree info */
extern float max_depth;
extern int leaf_num;

/* Progress record out of 1024 */
extern int progress;

/* Time record */
extern struct timeval start,finish, last_finish,start1,finish1;

/* If the input range is less than ADV_THRESHOLD, 
then it will check for concrete adversarial example*/
#define ADV_THRESHOLD  0.00001

/* Thread locker */
pthread_mutex_t lock;

/* Active threads number */
extern int count;



struct check_property_by_forward_backward_refining_args
{
    struct NNet *network;
    struct Interval *input;
		 float **value_upper;
 float **value_lower;
		 float **symbol_upper;
 float **symbol_lower;
 int **states;
 float *equation_upper;
float *equation_lower;
 float *new_equation_upper;
float *new_equation_lower;
		 lprec *lp;
 int *rule_num;
int property;
int need_prop;
int depth;
int split_layer;
int change_split_layer;
int input_refined;
};


/*
 * Check the concrete adversarial examples of 
 * the middle point of given input ranges.
 */
void check_adv(struct NNet *nnet, struct Interval *input);


/*
 * Check the predefined properties given 
 * approximated output ranges.
 */
int check_functions(struct NNet *nnet, struct Interval *output);

int check_functions_wit_constraints(struct NNet *nnet, struct Interval *output,float **symbol_upper, float **symbol_lower,lprec *lp);

/*
 * Check the predefined properties given 
 * concrete output.
 */
int check_functions1(struct NNet *nnet, struct Matrix *output);



#endif
