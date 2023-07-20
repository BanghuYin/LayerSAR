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

extern int NEED_PRINT;

extern int DEBUG;
extern int can_t_prove;

extern int adv_found;
extern int forward_count;

/* Mode for faster search concrete
adversarial examples */
extern int CHECK_ADV_MODE;


/* Time record */
extern struct timeval start,finish, last_finish,start1,finish1;


/* Thread locker */
pthread_mutex_t lock;

/* Active threads number */
extern int count;

struct check_property_by_forward_backward_refining_args
{
    struct NNet *network;
    struct Interval *input;
struct Interval *output;
		 float **value_upper;
 float **value_lower;
		 float **symbol_upper;
 float **symbol_lower;
 int **states;
		 lprec *lp;
 int *rule_num;
int property;
int need_prop;
int depth;
int split_layer;
int change_split_layer;
int input_refined;
int *output_map;
};



#endif
