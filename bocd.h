#ifndef BOCD_H
#define BOCD_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>

namespace bocd {

class StudentT {
public:
  StudentT(double alpha, double beta, double kappa, double mu);

  xt::xarray<double> pdf(double data);

  void update_theta(double data);

private:
  xt::xarray<double> m_alpha0;
  xt::xarray<double> m_beta0;
  xt::xarray<double> m_kappa0;
  xt::xarray<double> m_mu0;

  xt::xarray<double> m_alpha;
  xt::xarray<double> m_beta;
  xt::xarray<double> m_kappa;
  xt::xarray<double> m_mu;
};

class ConstantHazard {
public:
  ConstantHazard(int lam):m_lam(lam){}
  xt::xarray<double> operator()(int shape) {
    return 1.0/m_lam * xt::ones<double>({shape,});
  }
private:
  int m_lam;
};

class BOCD {

public:
  BOCD(int lam, double alpha, double beta, double kappa, double mu);
  void update(double data);

  size_t m_t0;
  size_t m_t;
  xt::xarray<double> m_growth_probs;
  StudentT m_observation_likelihood;
  ConstantHazard m_hazard_function;
};
}

#endif