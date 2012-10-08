#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"

class l2r_l2_mmdt_fun: public function
{
public:
	l2r_l2_mmdt_fun(const problem *prob, const model *hyperplane_model, double *C);
	~l2r_l2_mmdt_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
  double *tmp;
  double *hyperplaneCorr;

	int *I;
	int sizeI;
	const problem *prob;
  const model *hyperplane_model;
};

l2r_l2_mmdt_fun::l2r_l2_mmdt_fun(const problem *prob, const model *hyperplane_model, double *C)
{
	int l=prob->l;

	this->prob = prob;
  this->hyperplane_model = hyperplane_model;

	z = new double[l];
	D = new double[l];
	I = new int[l];

  int nr_class = hyperplane_model->w;
  hyperplaneCorr = new double [ nr_class * nr_class ];
  for ( int k1 = 0 ; k1 < nr_class ; k1++ )
    for ( int k2 = 0 ; k2 < nr_class ; k2++ )
    {
      hyperplaneCorr[ k1 + nr_class*k2 ] = 0;
      for ( int d = 0 ; d < orig_dim ; d++ )
      {
        hyperplaneCorr[ k1 + nr_class * k2 ] += hyperplane_model->w[ k1 * nr_class + d ] *
          hyperplane_model->w[ k2 * nr_class + d ];
      }
    }

  tmp = new double[ get_nr_variable() ];
	this->C = C;
}

l2r_l2_mmdt_fun::~l2r_l2_mmdt_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;

  delete[] tmp;
  delete[] hyerplaneCorr;
}

double calcReg ( double *w )
{
  int original_dimension = hyperplane_model->nr_feature;
  int number_of_classes = hyperplane->nr_class;
	int current_dimension = get_nr_variable();

  // FIXME: exploit symmetry
  double reg = 0.0;
  // complexity of this step: O(K*K*D)
  // K is the number of classes
  // D is the source dimension
  for ( int k1 = 0 ; k1 < number_of_classes ; k1++ )
    for ( int k2 = 0 ; k2 < number_of_classes ; k2++ )
    {
      double beta_sp = 0.0;
      for ( int d = 0 ; d < original_dimension ; d++ )
        beta_sp += w[ k1 * original_dimension + d ] * w[ k2 * original_dimension + d ];

      reg += hyperplaneCorr[ k1*number_of_classes + k2 ] * beta_sp;
    }
      

  return 0.5 * reg;
}

/** */
void multReg ( double *w, double *tmp )
{
  // dimension of the source domain
  int original_dimension = hyperplane_model->nr_feature;
  // number of classes in the source domain
  int number_of_classes = hyperplane->nr_class;
  // the current dimension should be original_dimension * number_of_classes
  // this is the dimension of the augmented dataset
	int current_dimension = get_nr_variable();

  if ( current_dimension != number_of_classes * original_dimension )
  {
    fprintf(stderr, "dimension mismatch %d does not equal to %d x %d\n", current_dimension, number_of_classes, original_dimension);
    return;
  }

  int i = 0;
  int taski, taskj;
  
  for ( int i=0; i<current_dimension; i++ )
  {
    tmp[i]=0;
    for ( int j=0; j<current_dimension; j++ )
    {
      taski = i / original_dimension;
      taskj = j / original_dimension;
      tmp[i] += w[j] * hyperplaneCorr[ taski*number_of_classes + taskj ]; 
    }
  }

}

/** compute the objective function */
double l2r_l2_mmdt_fun::fun(double *w)
{
  // the objective function
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

  // old regularization code
	//for(i=0;i<w_size;i++)
	//	f += w[i]*w[i];
	//f /= 2.0;

  // compute the regularization term
  f = calcReg(w);

  // compute the hinge loss terms
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_mmdt_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

  // old code for standard regularization
	//for(i=0;i<w_size;i++)
	//	g[i] = w[i] + 2*g[i];

  // compute new regularization term
  multReg(w, g);

  for(i=0;i<w_size;i++)
		g[i] += 2*g[i];

}

int l2r_l2_mmdt_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_mmdt_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_mmdt_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_mmdt_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_mmdt_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

