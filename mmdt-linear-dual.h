#ifndef MMDTLINEARDUAL_INCLUDE
#define MMDTLINEARDUAL_INCLUDE

#include "linear.h"

enum {
  VERBOSELEVEL_SILENT = 0,
  VERBOSELEVEL_NORMAL = 1,
  VERBOSELEVEL_DETAIL = 2,
  VERBOSELEVEL_DEBUG  = 5
};


typedef struct mmdt_parameter {
  bool explicitW;
  int max_iterations;
  double eps;
  double epsW;
  bool calculate_hinge;
  bool warm_start_W;
  bool warm_start_theta;
  bool regularize_identity;
  int verbose_level;
  int max_iter;
  float ratio_active_size;
  bool return_transform_w;
} mmdt_parameter;

typedef struct mmdt_model {
  double *w_transform;
  double *alpha;
  double *beta;
  bool beta_available;
} mmdt_model;

/**
* @brief MMDT Combined Solver (optimizes hyperplanes and transformation)
*
* @param prob_target target training data (liblinear data struct)
* @param prob_source source training data (liblinear data struct)
* @param w_transform pointer to the resulting transformation matrix W
* @param param liblinear parameters for hyperplane optimization 
* @param mmdt_param liblinear parameters for transformation optimization
*
* @return liblinear model for the target domain 
*/
model *mmdt_solver ( const problem *prob_target, const problem *prob_source, double *w_transform, const struct parameter *param, const struct mmdt_parameter *mmdt_param );


/**
* @brief Perform optimization with respect to the transformation
*
* @param prob target training data
* @param hyperplane_model previously estimated hyperparameters in the source domain
* @param transform_model resulting MMDT model including the parameters of the transformation (has to be pre-initialized!)
* @param eps accuracy setting of the liblinear optimizer
* @param solver_type only l2-loss and l1-loss is supported (have a look in the liblinear code)
* @param explicitW if set to true, we will not use dual variables beta to speed up the optimization when the number of tasks is less than the dimension of the features
* @param warm_start if set to true, previous values in transform_model will be used
*/
void solve_l2r_l1l2_mmdt(
	const problem *prob, 
  const model *hyperplane_model, 
  struct mmdt_model *transform_model, 
  double eps, 
  int solver_type, 
  bool explicitW = true, 
  bool warm_start = false, 
  bool regularize_identity = false,
  int max_iter = 1000,
  float ratio_active_size = 1.0 );

/* ------------- utility functions ----------- */
/**
* @brief Fill parameter structure with standard default values.
*
* @param mmdt_param pointer to the structure to be filled.
*/
void std_mmdt_parameter ( struct mmdt_parameter *mmdt_param );

/**
* @brief initialize the vectors of the MMDT model and set all the parameters to zero
*
* @param transform_model MMDT model structure
* @param nr_class number of classes/tasks
* @param dim_t target domain dimension
* @param dim_s source domain dimension
* @param nt number of examples in the target domain
* @param explicitW initialize the second type of dual variables
* @param initialize_w_transform initialize the W transformation pointer?
*/
void initialize_mmdt_model ( struct mmdt_model *transform_model, int nr_class, int dim_t, int dim_s, int nt, bool explicitW, bool intialize_w_transform = true ); 

/**
* @brief free the elements of the MMDT structure
*
* @param transform_model
*/
void free_mmdt_model ( struct mmdt_model *transform_model );

#endif
