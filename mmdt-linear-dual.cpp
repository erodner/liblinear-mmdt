#ifndef MMDTLINEAR_DUAL
#define MMDTLINEAR_DUAL

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>

#include <set>

#include "linear.h"
#include "tron.h"

#include "linear.h"

// same standard typedefs and macros
#include "shared.h"

#include "mmdt-linear-dual.h"


#define COMPARE_WITH_MATLAB_VERSION
// supporting weights
#define GETI(i) (i)

static int verbose_level = VERBOSELEVEL_NORMAL;



/** objective function of the dual MMDT problem */
double fun_l2r_l1l2_mmdt ( double *w, int w_size, double *alpha, int l, double *diag, int &nSV )
{
	double v = 0;
	nSV = 0;
  int i;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
  return v/2;
}

/** optimization with respect to the transformation W */
void solve_l2r_l1l2_mmdt(
	const problem *prob, // the whole target data
  const model *hyperplane_model, // hyperplanes
  struct mmdt_model *transform_model, // model containing the W estimation
  double eps, // demanded accuracy and termination criterion
  int solver_type, 
  bool explicitW, // dual or primal representation of W
  bool warm_start, // should we use a former estimate of the transformation?
  bool regularize_identity, // use \| W - I \|^2 regularization?
  int max_iter,  // this is the maximum number of iterations 
                 // an iteration is one loop through the whole active set
  float ratio_active_size  // relative size of the active set
                           // this can be used to significantly reduce the computation time,
                           // however, it might be critical for the overall performance and convergence
) 
{
  // category pairs with a hyperplane correlation below a certain threshold are simply ignored
  // this could speed up the algorithm in case of "independent" category models
  const double hyperplane_correlation_threshold = 1e-9;

  bool use_caching = ! explicitW;

  int stepcount = 0;

  int l_features = prob->l;
  // target dimension
  int dim_t = prob->n;
  // source dimension
  int dim_s = hyperplane_model->nr_feature;
  // number of classes
  int nr_class = hyperplane_model->nr_class;

  // we don't really need these positive and negative C values in our setting
  const double Cp = 1.0;
  const double Cn = 1.0;

	int l = l_features * nr_class;

  if ( verbose_level >= VERBOSELEVEL_DETAIL ) {
    info("[MMDT Solver] number of augmented features = %d\n", l );
    info("[MMDT Solver] number of original features = %d\n", l_features );
    info("[MMDT Solver] number of classes = %d\n", nr_class);  
  }

  // the dimension of the W matrix
  double *w = transform_model->w_transform;
  double *beta = transform_model->beta;
	double *alpha = transform_model->alpha;

  int w_size = dim_t * dim_s;
  int beta_size = nr_class * dim_t;  

  if ( verbose_level >= VERBOSELEVEL_DETAIL ) {
    info("[MMDT Solver] total size of the parameter vector is %d\n", explicitW ? w_size : beta_size);
    info("[MMDT Solver] dimension of the source domain = %d\n", dim_s);  
    info("[MMDT Solver] dimension of the target domain = %d\n", dim_t);  
  }
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int *index = new int[l];
  int *y = new int[l];

  // this is the number of coordinates descended in each iteration
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double *diag = new double[l];
	double *upper_bound = new double[l];
	double *C_ = new double[l];

  // some macros for calculating indices of augmented features
  // k task/class index
  // j example index (non-augmented)
  // i augmented example index
  #define INDEX_EXAMPLE_TASK(i,j,k) int j = (i) % l_features; int k = (i) / l_features;
  #define EXAMPLE_TASK_INDEX(i,j,k) int i = (k)*l_features + (j);
  #define ISCLASS(j,k) ( (prob->y[j] == hyperplane_model->label[k]) ? 1 : -1 )

  // RUNTIME: O(N*M)
	for(i=0; i<l; i++) 
	{
    INDEX_EXAMPLE_TASK(i,j,k);
		if(ISCLASS(j,k) == 1)
			C_[i] = prob->W[j] * Cp;
		else 
			C_[i] = prob->W[j] * Cn;
		diag[i] = 0.5/C_[i];
		upper_bound[i] = INF;
	}
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		for(i=0; i<l; i++) 
		{
			diag[i] = 0;
			upper_bound[i] = C_[i];
		}
	}

  // RUNTIME: O(N*M)
	for(i=0; i<l; i++)
	{
    INDEX_EXAMPLE_TASK(i,j,k);
    y[i] = ISCLASS(j,k);
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
  // 
  // Original code: set the initial values to zero
  //
  if ( ! warm_start )
  {
    for(i=0; i<l; i++)
    	alpha[i] = 0;
    for(i=0; i<w_size; i++)
    	w[i] = 0;
    if ( ! explicitW ) 
      for(i=0; i<beta_size; i++)
        beta[i] = 0.0;
  }


  // The new code is based on the following derivation
  // w_k = sum_i y_i alpha_i d^i_k
  //   w = sum_i y_i alpha_i d^i
  //     = sum_i y_i alpha_i theta^r ( x^j )^T
  //     = sum_j sum_r [ y_j == r ] alpha_i theta^r ( x^j )^T
  //     = sum_r theta^r ( sum_j [ y_j == r ] alpha_i ( x^j )^T
  //     = sum_r theta^r \beta^r
  // This step can be skipped in the case \beta is zero and we do not
  // do a warm start. However, when \theta^r is changing we should re-calculate w,
  // which is probably the case when a warm start is done. 
  // But keep in mind: we only have to do that when w is directly used within the optimization
  if ( warm_start )
  {
    // take care of consistency between w and beta (explicitW=false) or w and alpha (explicitW=true)
    // by fixing the alpha vector and re-calculating w and beta
    for ( int l = 0 ; l < dim_t * dim_s; l++ )
      w[l] = 0.0;

    double *betavec = NULL;
    if ( explicitW )
      betavec = new double [ dim_t ];

    // RUNTIME: O(M*N*D) + O(M*Ds*Dt) (warm_start=1)
    for ( int k = 0 ; k < nr_class ; k++ )
    {
      if ( explicitW ) 
      {
        for ( int dt = 0 ; dt < dim_t ; dt++ )
          betavec[dt] = 0.0;
        for ( int j = 0 ; j < l_features; j++ )
        {
          EXAMPLE_TASK_INDEX(i,j,k);
          double alphay = alpha[i] * y[i];

          feature_node *xj = prob->x[j];
          while(xj->index!= -1)
          {
            betavec[ xj->index - 1 ] += alphay * (xj->value);
            xj++;
          }
        }
      } else {
        betavec = beta + k * dim_t;
      }

      // re-calculate w
      for ( int dt = 0 ; dt < dim_t ; dt++ )
        for ( int ds = 0 ; ds < dim_s ; ds++ )
          w[ ds * dim_t + dt ] += betavec[ dt ] * hyperplane_model->w[ ds * nr_class + k ];
    }

    if ( explicitW )
      delete [] betavec;
  }

  //
  // Calculation of the diagonal matrix entries QD
  //   QD_i = sum_k (d^i_k)^2 + diag_i
  //        = tr ( x^j * (theta^r)^T * theta^r * x^j^T ) + diag_i
  //        = | x^j |^2 | theta^r |^2 + diag_i

  // RUNTIME: O(M^2) time complexity
  double *hyperplane_norm = new double[ nr_class ];
  for ( int k = 0; k < nr_class ; k++ )
  {
    hyperplane_norm[k] = 0.0;
    for ( int ds = 0; ds < dim_s; ds++ )
    {
      double wvalue = hyperplane_model->w[ ds*nr_class + k ];

      hyperplane_norm[k] += wvalue * wvalue;
    }
  }

  double *hyperplane_correlation = NULL;
  if ( ! explicitW )
  {
    if ( verbose_level >= VERBOSELEVEL_DETAIL )
      info("[MMDT Solver] Computing hyperplane correlations...\n");
    // RUNTIME: O(M^2) (explicitW=0)
    hyperplane_correlation = new double [ nr_class * nr_class ];
    for ( int k1 = 0; k1 < nr_class; k1++ )
    {
      hyperplane_correlation[ k1 * nr_class + k1 ] = hyperplane_norm[k1];
      for ( int k2 = k1+1; k2 < nr_class; k2++ )
      {
        double sp = 0.0;
        for ( int ds = 0; ds < dim_s; ds++ )
          sp += hyperplane_model->w[ ds*nr_class + k1 ] * hyperplane_model->w[ ds*nr_class + k2 ];

        hyperplane_correlation[ k1 * nr_class + k2 ] = sp;
        hyperplane_correlation[ k2 * nr_class + k1 ] = sp;
      }
    }
  }

  // RUNTIME: O(N D) + O(N M)
  if ( verbose_level >= VERBOSELEVEL_DETAIL )
    info("[MMDT Solver] Calculating diagonal matrix Q...\n");
  for ( int j = 0 ; j < l_features; j++ )
  {
    double feature_norm = 0.0;

		feature_node *xj = prob->x[j];
		while (xj->index != -1)
		{
      // calculation of the scalar product
			double val = xj->value;
      feature_norm += val*val;
			xj++;
    }

    for ( int k = 0 ; k < nr_class ; k++ )
    {
		  EXAMPLE_TASK_INDEX(i,j,k);

      QD[i] = diag[GETI(i)] + feature_norm * hyperplane_norm[k];
      index[i] = i;
    }
  }
  
  // old code of the original solver
	/*
  for(i=0; i<l; i++)
		QD[i] = diag[GETI(i)];
		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += y[i]*alpha[i]*val;
			xi++;
		}
		index[i] = i;
	}
  */

  if ( verbose_level >= VERBOSELEVEL_DETAIL )
    info("[MMDT Solver] Optimization ...\n");

  // timestamps that indicate when beta vectors have been updated
  int *beta_cache_t = NULL;
  // timestamps that indicate when beta vector scalar products have been updated
  int *betasp_cache_t = NULL;
  double *betasp_cache = NULL;

  if ( use_caching )
  {
    // allocate and initialize caching
    beta_cache_t = new int[ nr_class ];
    betasp_cache_t = new int[ l ];
    betasp_cache = new double [ l ];

    memset( beta_cache_t, 0, nr_class * sizeof(int) );
    memset( betasp_cache_t, 0, l * sizeof(int) );
    memset( betasp_cache, 0, l * sizeof(double) );
  }


	while (iter < max_iter)
	{
    // this might take a long time, but it is good for debugging
    /*
    if ( verbose_level >= VERBOSELEVEL_DEBUG )
      if ( iter % 5 == 0 )
      {
        int nSV = 0;
        info("[%d] objective value = %lf\n", iter, fun_l2r_l1l2_mmdt ( w, w_size, alpha, l, diag, nSV ) );
      }
    */

		PGmax_new = -INF;
		PGmin_new = INF;

    if ( verbose_level >= VERBOSELEVEL_DETAIL )
      info("[MMDT Solver] Iteration %d: active size = %d\n", iter, active_size);

    if ( use_caching )
    {
      // only shuffle within one class
      for (i=0; i<active_size; i++)
      {
        INDEX_EXAMPLE_TASK(i,j,k);
        int nright = (k+1) * l_features - i;
        int ii = i + rand() % nright;
        swap(index[i], index[ii]);
      }
    } else {
      // shuffeling the active set represented by index
      for (i=0; i<active_size; i++)
      {
        int j = i+rand()%(active_size-i);
        swap(index[i], index[j]);
      }
    }
    
    // looping through some examples and doing a coordinate descent step
    int maximum_active_size = (int)(ratio_active_size*l);
		for (s=0; s< std::min( active_size, maximum_active_size ); s++)
		{
			i = index[s];

      // G will be the gradient with respect to i'th dual variable
			G = 0;
			schar yi = y[i];

      INDEX_EXAMPLE_TASK(i,j,k);
      
      // New code for the calculation of the scalar product of w and an example
      //
      if ( explicitW ) 
      {
        // G = w^T d^i = theta_k^T * W * x^j
        // RUNTIME: O(D^2) (explicitW=1) for each iteation
        for ( int ds = 0 ; ds < dim_s ; ds++ )
        {
          double transformed_xj_ds = 0.0;

          feature_node *xj = prob->x[j];
          while(xj->index!= -1)
          {
            transformed_xj_ds += w[ ds * dim_t + xj->index-1 ] * (xj->value);
            xj++;
          }

          G += transformed_xj_ds * hyperplane_model->w[ ds*nr_class + k ];
        }
      } else { 
        // G = w^T d^i = theta_k^T * W * x^j = theta_k^T sum_k' theta_k' beta_k'^T x^j
        // RUNTIME O(D * M) (explicitW=0) for each iteration
        double *hcp = hyperplane_correlation + k * nr_class;
        for ( int k2 = 0 ; k2 < nr_class ; k2++,hcp++ )
        {
          double hc = (*hcp);

          // if the hyperplane correlation is below a threshold, there is no point 
          // in calculating the scalar product
          if ( fabs(hc) < hyperplane_correlation_threshold ) continue;

          double sp = 0.0;
          
          if ( use_caching )
          {
            int i2 = k2 * l_features + j;
            if ( beta_cache_t[k2] < betasp_cache_t[i2] ) 
            {
              sp = betasp_cache[i2];
            } else {
              feature_node *xj = prob->x[j];
              double *betavec = beta + k2*dim_t;
              while(xj->index!= -1)
              {
                sp += betavec[ xj->index-1 ] * (xj->value);
                xj++;
              }
              betasp_cache[i2] = sp;
              betasp_cache_t[i2] = stepcount;
            }
          } else {
            feature_node *xj = prob->x[j];
            double *betavec = beta + k2*dim_t;
            while(xj->index!= -1)
            {
              sp += betavec[ xj->index-1 ] * (xj->value);
              xj++;
            }
          }
          G += sp * hc;
        }

      }
      // Old code of the original solver with non-augmented features
			/*
      feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{ G += w[xi->index-1]*(xi->value);
				xi++; }
      */

			G = G*yi-1;

      // if we are regularizing with respect to |W-I|
      // the w variable represents W-I, and vectors are indirectly transformed
      // with mat{w}+I inside the loss, this adds the following term to the gradient
      if ( regularize_identity )
      {
        feature_node *xj = prob->x[j];
        // RUNTIME O(D) (regularize_identity=1) for each iteration
        while(xj->index!= -1)
        {
          if ( xj->index <= dim_s )
            G -= hyperplane_model->w[ (xj->index-1)*nr_class + k ] * xj->value;
          xj++;
        }
      }

			C = upper_bound[GETI(i)];

      // now we have the final gradient
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
          // remove the example from the active set
					active_size--;
          swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
          // remove the example from the active set
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi; // dalpha_i
        
        // New code for applying the change to the parameter vector
        // w_k = dalpha_i d^i_k
        //   w = dalpha_i d^i
        //     = dalpha_i theta^r ( x^j )^T
        // We also exploit sparsity in x^j
        INDEX_EXAMPLE_TASK(i,j,k);
				feature_node *xj = prob->x[j];
        //
        if ( explicitW )
        {
          // RUNTIME: O(D^2) (explicitW=1) for each iteration
          while (xj->index != -1)
          {
            double *wp = w + xj->index-1;
            double *hp = hyperplane_model->w + k;
            double v = d * xj->value;
            for (int ds = 0; ds < dim_s;  ds++, wp+=dim_t, hp+=nr_class)
              (*wp) += v * (*hp);
              //w[ds * dim_t + xj->index-1] += d * xj->value * hyperplane_model->w[ds*nr_class + k];
            xj++;
          }
        } else {
          // RUNTIME: O(D) (explicitW=0) for each iteration
          double *betavec = beta + k * dim_t;
          while (xj->index != -1)
          {
            //beta[k * dim_t + xj->index-1] += d * xj->value;
            betavec[ xj->index-1 ] += d * xj->value;
            xj++;
          }

          if ( use_caching )
            beta_cache_t[k] = stepcount;
        }


        // Old code of the original solver using non-augmented features
        /*
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
        */
			}

      stepcount++;
		}

		iter++;
    if ( verbose_level >= VERBOSELEVEL_NORMAL )
  		if(iter % 10 == 0)
	  		info(".");

    if ( verbose_level >= VERBOSELEVEL_DEBUG )
      info("current eps = %f (limit is %f)\n", PGmax_new - PGmin_new, eps); 

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
        if ( verbose_level >= VERBOSELEVEL_NORMAL )
				  info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

  if ( verbose_level >= VERBOSELEVEL_NORMAL )
	  info("\n[MMDT Solver] transformation optimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
    if ( verbose_level >= VERBOSELEVEL_NORMAL )
		  info("\nWARNING: reaching max number of iterations.\nUsing another solver may be faster.\n\n");

  // It might be more efficient, to just use the beta vectors in general (also for future transformations)
  // But sometimes, having w directly could be beneficial
  if ( ! explicitW )
  {
    double *wp = w;
    for ( int i = 0; i < dim_t*dim_s; i++,wp++ )
      *wp = 0;

    wp = w;
    for ( int ds = 0; ds < dim_s; ds++ )
    {
      for ( int dt = 0; dt < dim_t; dt++,wp++ )
      {
        double *hp = hyperplane_model->w + ds * nr_class;
        double *bp = beta + dt;
        for ( int k = 0; k < nr_class ; k++, hp++,bp+=dim_t )
          *wp += (*hp) * (*bp);
      }
    }

    for ( int i = 1; i < 10 ; i++ )
      info("W Matrix %f\n", w[i] );
  }

	// calculate objective value
  if ( verbose_level >= VERBOSELEVEL_DETAIL )
  {
	  int nSV = 0;
  	info("[MMDT Solver] transformation objective value = %lf\n", fun_l2r_l1l2_mmdt ( w, w_size, alpha, l, diag, nSV ) );
	  info("[MMDT Solver] nSV = %d\n",nSV);
  }

  // clean up all the data structures we used
  if ( hyperplane_correlation != NULL )
    delete [] hyperplane_correlation;

  if ( use_caching )
  {
    delete [] betasp_cache_t;
    delete [] beta_cache_t;
    delete [] betasp_cache;
  }

  delete [] hyperplane_norm;
  delete [] upper_bound;
	delete [] diag;
	delete [] C_;
	delete [] QD;
	delete [] y;
	delete [] index;
}


double hinge_loss( const model *m, const problem *prob )
{
  double sum = 0.0;
  for ( int k = 0; k < m->nr_class ; k++ )
    for ( int i = 0; i < prob->l; i++ )
    {
      // weight of the instance
      double weight = prob->W[i];
      double deltaTerm = ( prob->y[i] == m->label[k] ) ? 1 : -1;
      // scalar product with the hyperplane
      double score = 0.0; 

      feature_node *xi = prob->x[i];
      while ( xi->index != -1 )
      {
        score += xi->value * m->w[ (xi->index-1) * m->nr_class + k ];
        xi++;
      }
      double z = 1 - deltaTerm * score;
      double h =  ( z < 0.0 ? 0.0 : z );
      // L2-loss 
      sum += weight * h * h;
    }

  return sum;
}

double detailed_status ( int iteration, struct problem *prob_combined, struct model * hyperplane_model, double *w_transform,
  int dimt, int dims, bool regularize_identity = false )
{
  if ( verbose_level > VERBOSELEVEL_DETAIL )
    info("[MMDT Solver] Calculating the objective function...\n");

  double reg_hyperplane = 0.0;
  double reg_transformation = 0.0;
  double hinge = 0.0;

  for ( int k = 0; k < hyperplane_model->nr_class * dims ; k++ )
    reg_hyperplane += hyperplane_model->w[k] *  hyperplane_model->w[k];

  hinge = hinge_loss( hyperplane_model, prob_combined );
  
  // compute the regularization term of the transformation matrix
  for ( int k = 0; k < dims*dimt; k++ )
    reg_transformation += w_transform[k]*w_transform[k];

  double mmdt_objval = 0.5*reg_transformation + 0.5*reg_hyperplane + hinge;

  if ( verbose_level >= VERBOSELEVEL_NORMAL )
    info("[MMDT Solver] (iteration %d) f=%f (%f %f %f)\n", iteration, mmdt_objval, hinge, reg_hyperplane, reg_transformation );

  return mmdt_objval;
}


void force_consistency ( const problem *prob, int nr_task, const int *labels )
{
  if ( prob->w_in == NULL || prob->alpha_in == NULL || labels == NULL )
    return;

  int l = prob->l;
  int dim = prob->n;
  // re-calculate w based on alpha
  for ( int k = 0 ; k < nr_task ; k++ )
  {
    double *w = prob->w_in + dim*k;
    double *alpha = prob->alpha_in + l*k;
    int classk = labels[k];

    //for ( int tt=0; tt < 5 ; tt++ )
    //  info("%f ", w[tt] );
    //info(" before (%d)\n", classk);

    for ( int d = 0 ; d < dim ; d++ )
      w[d] = 0.0;

    for ( int j = 0 ; j < l ; j++ )
    {
      double alphay = alpha[j] * ( (prob->y[j] == classk) ? 1.0 : -1.0 );
      feature_node *xj = prob->x[j];
      while ( xj->index != -1 )
      {
        w[ xj->index - 1 ] += alphay * xj->value;
        xj++;
      }
    }
    
    //for ( int tt=0; tt < 5 ; tt++ )
    //  info("%f ", w[tt] );
    //info(" after\n");

  }
}



// combined mmdt solver
model *mmdt_solver ( const problem *prob_target, const problem *prob_source, double *w_transform, const struct parameter *param, const struct mmdt_parameter *mmdt_param )
{
  // calculating the hinge loss needs some time, but it is good for debugging purposes and flexible termination criteria
  bool calculate_hinge = mmdt_param->calculate_hinge;
  bool regularize_identity = mmdt_param->regularize_identity;

  // setting global verbose level
  verbose_level = mmdt_param->verbose_level;
  // switch off status reports of liblinear itself
  if ( verbose_level <= VERBOSELEVEL_NORMAL )
    set_print_string_function ( &print_string_null );

  // dimensionality of source and target domain
  int dims = prob_source->n;
  int dimt = prob_target->n;
  // determine the number of classes: using the STL is so convenient here
  // and we only have to do it once
  std::set<double> classesA;
  for ( int j = 0; j < prob_target->l ; j++ )
    classesA.insert( prob_target->y[j] );
  for ( int j = 0; j < prob_source->l ; j++ )
    classesA.insert( prob_source->y[j] );
  int nr_class = classesA.size(); // number of target classes 

  // combined problem = source problem + transformed target problem
  problem *prob_combined = Malloc(problem, 1);
  // the combined problem has the same dimension
  prob_combined->n = prob_source->n;
  // the size is added by the number of training examples from the target domain
  prob_combined->l = prob_source->l + prob_target->l;
  // the label vector
  prob_combined->y = Malloc(double, prob_combined->l);
  memcpy( prob_combined->y, prob_source->y, sizeof(double)*prob_source->l );
  double *y_target_labels = prob_combined->y + prob_source->l;
  memcpy( y_target_labels, prob_target->y, sizeof(double)*prob_target->l );
  // bias (no bias source at all, take care for it yourself!)
  prob_combined->bias = -1;
  // weights (we copy the weights instead of 
  prob_combined->W = Malloc(double, prob_combined->l);
  memcpy( prob_combined->W, prob_source->W, sizeof(double)*prob_source->l );
  double *W_target = prob_combined->W + prob_source->l;
  memcpy( W_target, prob_target->W, sizeof(double)*prob_target->l );
  // preparing data pointers
  prob_combined->x = Malloc( struct feature_node *, prob_combined->l );
  // copy the pointer to the source dataset
  memcpy( prob_combined->x, prob_source->x, sizeof( struct feature_node * )*prob_source->l);
  // preparing some space for transformed target examples
  // the next allocation is somehow critical, because those features do not have to be sparse at all, therefore, allocation can be really critical here
  // one idea is to do a transformation of hyperplanes within the hyperplane optimization part
  if ( verbose_level >= VERBOSELEVEL_DETAIL )
    info("[MMDT Solver] Allocating %d vectors of size %d+1 for transformed target examples\n",  prob_target->l, prob_source->n );

  // setting up the combined problem with source and transformed target examples
  feature_node *transformed_target_examples = Malloc(feature_node, prob_target->l * (prob_source->n+1) );
  // set pointers
  for ( int j = 0; j < prob_target->l ; j++ )
  {
    prob_combined->x[ prob_source->l + j ] = transformed_target_examples + j*(prob_source->n+1);
    // set initial value to only the bias term
    feature_node *x = prob_combined->x[ prob_source->l + j ];
    x->value = 0.0; // does not matter at all :)
    x->index = -1; // end of sparse vector
  }
  // initialize the previous/first solution
  if ( mmdt_param->warm_start_theta )
  {
    prob_combined->alpha_in = Malloc(double, nr_class * prob_combined->l );
    for ( int k = 0; k < nr_class*prob_combined->l ; k++ )
      prob_combined->alpha_in[k] = 0.0;
    prob_combined->w_in = Malloc(double, nr_class * prob_combined->n);
    for ( int k = 0; k < nr_class*prob_combined->n ; k++ )
      prob_combined->w_in[k] = 0.0;
  } else {
    prob_combined->alpha_in = NULL;
    prob_combined->w_in = NULL;
  }
        
  // initialize the transformation
  for ( int k = 0; k < dims*dimt; k++ )
    w_transform[k] = 0.0;
 
  bool converged = false;
  double mmdt_objval = 0.0;
  double old_mmdt_objval = 0.0;
  int it = 0;

  // the model is initialized with the maximum number of parameters (with respect to the number of classes)
  // after the first SVM learning step
  struct mmdt_model transform_model;
  struct model *hyperplane_model;
  
  // use the coordinate descent approach
  while ( !converged && (it < mmdt_param->max_iterations) )
  {
    // (1) learn the transformation with fixed hyperplane parameters
#ifdef COMPARE_WITH_MATLAB_VERSION
    srand(1);
#endif
    if ( it > 0 )
    {
      if ( it == 1 )
      {
        initialize_mmdt_model ( &transform_model, nr_class, dimt, dims, prob_target->l, mmdt_param->explicitW, false /* do not initialize w_transform */);
        transform_model.w_transform = w_transform;
      }

      if ( verbose_level >= VERBOSELEVEL_NORMAL )
        info("[MMDT Solver] Estimating the transformation matrix...\n");
      // now we run the main solver
      solve_l2r_l1l2_mmdt( prob_target, 
                           hyperplane_model, 
                           &transform_model,
                           // parameters --------------
                           mmdt_param->epsW, 
                           param->solver_type, 
                           mmdt_param->explicitW, 
                           mmdt_param->warm_start_W /* warm start */, 
                           regularize_identity,
                           mmdt_param->max_iter,
                           mmdt_param->ratio_active_size);
    
      // (2.1) transform the target examples
      if ( verbose_level >= VERBOSELEVEL_DETAIL )
        info("[MMDT Solver] Transforming the target data ...\n");
      double *beta_sp = NULL;
      for ( int j = 0; j < prob_target->l ; j++ )
      {
        feature_node *transformed_vector = transformed_target_examples + (prob_source->n + 1) * j;
       
        if ( mmdt_param->explicitW )
        {
          // simple W*x multiplication
          for ( int ds = 0; ds < prob_source->n ; ds++ )
          {
            feature_node *orig_vector = prob_target->x[j];
            double sum = 0.0;
            while ( orig_vector->index != -1 )
            {
              sum += w_transform[ prob_target->n * ds + orig_vector->index - 1 ] * orig_vector->value;

              // check wheter this is a diagonal element of w_transform
              // if yes, we have to add the identity in the case of |W-I| regularization
              if ( regularize_identity && ( orig_vector->index - 1 == ds ) )
                sum += orig_vector->value;

              orig_vector++;
            }
            if ( sum != 0.0 )
            {
              transformed_vector->index = ds + 1;
              transformed_vector->value = sum;
              transformed_vector++;
            }
          }
        } else {
          // transforming the data can be speed-up in the case we represent W as the low rank matrix
          // in this case w is calculated as follows
          // for k=1..nr_tasks: w[ ds * dim_t + dt ] += hyperplane_model->w[ ds*nr_class + k ] * beta[ k * dim_t + dt ];
          if ( beta_sp == NULL )
            beta_sp = new double[ nr_class ];
          
          double *beta = transform_model.beta;

          for ( int k = 0 ; k < nr_class ; k++ )
          {
            feature_node *orig_vector = prob_target->x[j];
            
            double sum = 0.0;
            while ( orig_vector->index != -1 )
            {
              sum += beta[ k * dimt + orig_vector->index - 1 ] * orig_vector->value;
              orig_vector++;
            }
            beta_sp[k] = sum;
          }

          feature_node *ov = prob_target->x[j];
          for ( int ds = 0; ds < dims; ds++ )
          {
            double sum = 0.0;

            // we have to add the identity in the case of |W-I| regularization
            if ( regularize_identity )
            {
              // we assume that sparse vector indices are ordered
              if ( ov->index == ds+1 ) {
                sum += ov->value;
                ov++;
              }
            }

            for ( int k = 0 ; k < nr_class ; k++ )
            {
              // there is no point in calculating the remaining stuff
              if ( fabs(beta_sp[k]) < 1e-20 ) continue;
              
              sum += hyperplane_model->w[ ds*nr_class + k ] * beta_sp[k];
            }
                     
            if ( sum != 0.0 )
            {
              transformed_vector->index = ds + 1;
              transformed_vector->value = sum;
              transformed_vector++;
            }
          }
        }
        // end the vector
        transformed_vector->index = -1;
      }

      // delete temporary storage
      if ( beta_sp != NULL )
        delete [] beta_sp;

      /* Just for real debugging: evaluate the hinge loss again */
      if ( verbose_level >= VERBOSELEVEL_DETAIL )
        if ( calculate_hinge )
          detailed_status( it, prob_combined, hyperplane_model, w_transform, dimt, dims, regularize_identity );
 
      // for warm starts of theta we have to care about the consistency of w and alpha
      // however, the current implementation sets w (hyperplanes) according to alpha, which contradicts to the fact
      // that we are using w (hyperplane) when optimizing for W (transformation)
      if ( mmdt_param->warm_start_theta )
        force_consistency( prob_combined, hyperplane_model->nr_class, hyperplane_model->label );
     
      // the previous solution is stored in prob_combined (somehow a dirty hack, sorry :)
      free_and_destroy_model ( &hyperplane_model );
    }
   
    
    if ( verbose_level >= VERBOSELEVEL_NORMAL )
      info("[MMDT Solver] Estimating hyperplane parameters...\n");
        
#ifdef COMPARE_WITH_MATLAB_VERSION
    srand(1);
#endif

    // (2.2) solve with respect to the hyperplane parameters
    // liblinear: one-vs-all stuff and some nasty re-ordering
    hyperplane_model = train ( prob_combined, param );
    //force_consistency( prob_combined, hyperplane_model->nr_class, hyperplane_model->label );
    
    nr_class = hyperplane_model->nr_class;
    
    // calculate mmdt objective value (needs some time)
    if ( calculate_hinge ) {
      mmdt_objval = detailed_status( it, prob_combined, hyperplane_model, w_transform, dimt, dims, regularize_identity );
    } else {
      if ( verbose_level >= VERBOSELEVEL_NORMAL )
        info("[MMDT Solver] (iteration %d)\n", it );
    }

    // check for convergence in a very trivial manner
    if ( (it > 0) && (mmdt_objval > old_mmdt_objval) )
      converged = true;

    old_mmdt_objval = mmdt_objval;
    
    it++;
  }

  if ( verbose_level >= VERBOSELEVEL_NORMAL )
    info("[MMDT Solver] optimization done.\n");

  // if we regularize with respect to the identity, we switch the representation of w_transform here
  // by simply adding the identity
  if ( regularize_identity )
  {
    info("adding the final identity matrix ...\n");
    for ( int i = 0; i < 10 ; i++ )
      info("W Matrix(%d,%d) %f\n", i, i, w_transform[i*prob_target->n + i] );


    for ( int j = 0; j < std::min(prob_target->n, prob_source->n); j++ )
    {
      w_transform[ prob_target->n * j + j ] += 1.0;
    }
  } 
   for ( int i = 0; i < 10 ; i++ )
      info("W Matrix(%d,%d) %f\n", i, i, w_transform[i*prob_target->n + i] );

  // transform the hyperplane parameters
  // (this could be optimized as well)
  double *w_transformed = Malloc(double, nr_class * prob_target->n );
  for ( int k = 0 ; k < nr_class ; k++ )
  {
    for ( int dt = 0; dt < prob_target->n; dt++ )
    {
      double sum = 0.0;
      for ( int ds = 0; ds < prob_source->n; ds++ ) {
        sum += w_transform[ prob_target->n * ds + dt ] * hyperplane_model->w[ds*nr_class + k];
      }

      w_transformed[ dt*nr_class + k ] = sum;
    }
  }

  // substitute the hyperplane parameters in the source domain
  // with the ones in the target domain
  if ( mmdt_param->return_transform_w )
  {
    free( hyperplane_model->w );
    hyperplane_model->w = w_transformed;
  }

  // set the new size of the expected feature dimension
  hyperplane_model->nr_feature = prob_target->n;
  
  // clean up
  free( prob_combined->y );
  free( prob_combined->W );
  if ( prob_combined->alpha_in != NULL )
    free( prob_combined->alpha_in );
  if ( prob_combined->w_in != NULL )
    free( prob_combined->w_in );

  // free some huge memory size
  free( transformed_target_examples );

  // free nearly all model parameter
  transform_model.w_transform = NULL;
  free_mmdt_model( &transform_model );

  return hyperplane_model;
}


void std_mmdt_parameter ( struct mmdt_parameter *mmdt_param )
{
  mmdt_param->max_iterations = 4;
  mmdt_param->eps = 0.1;
  mmdt_param->epsW = 1.0;
  mmdt_param->calculate_hinge = true;
  mmdt_param->explicitW = false;
  mmdt_param->verbose_level = VERBOSELEVEL_NORMAL;
  mmdt_param->warm_start_W = false;
  mmdt_param->warm_start_theta = false;
  mmdt_param->max_iter = 1000;
  mmdt_param->ratio_active_size = 1.0;
  mmdt_param->return_transform_w = true;
  
  mmdt_param->regularize_identity = true;
}

void initialize_mmdt_model ( struct mmdt_model *transform_model, int nr_class, int dim_t, int dim_s, int nt, bool explicitW, bool initialize_w_transform )
{
  if ( initialize_w_transform ) {
    // transformation matrix itself
    transform_model->w_transform = Malloc(double, dim_t * dim_s);
    for ( int i = 0 ; i < dim_t * dim_s ; i++ )
      transform_model->w_transform[i] = 0.0;
  }
  // dual variables 1: one variable for each augmented features: nr_class * nt
  transform_model->alpha = Malloc(double, nr_class * nt);
  for ( int i = 0 ; i < nr_class * nt ; i++ )
    transform_model->alpha[i] = 0.0;
 
  transform_model->beta_available = !explicitW;
  if ( ! explicitW ) {
    // dual variables 2: W = sum_k \theta_k \beta_k^T
    transform_model->beta = Malloc(double, dim_t * nr_class);
    for ( int i = 0 ; i < dim_t * nr_class ; i++ )
      transform_model->beta[i] = 0.0;
  } else {
    transform_model->beta = NULL;
  }

}

void free_mmdt_model ( struct mmdt_model *transform_model )
{
  if ( transform_model->w_transform != NULL )
    free ( transform_model->w_transform );
  if ( transform_model->alpha != NULL )
    free ( transform_model->alpha );
  if ( transform_model->beta != NULL )
    free ( transform_model->beta );
}

#endif
