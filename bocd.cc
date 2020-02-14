#include <boost/math/distributions/students_t.hpp>
#include "bocd.h"

using namespace bocd;

StudentT::StudentT(double alpha, double beta, double kappa, double mu):
  m_alpha0({alpha}), m_alpha({alpha}), m_beta0({beta}), m_beta({beta}),
  m_kappa0({kappa}), m_kappa({kappa}), m_mu0({mu}), m_mu({mu}) {
}

xt::xarray<double> StudentT::pdf(double data) {
  xt::xarray<double> df = 2*m_alpha;
  xt::xarray<double> scale = xt::sqrt(m_beta*(m_kappa+1)/(m_alpha*m_kappa));
  xt::xarray<double> x = (data - m_mu) / scale;

  std::vector<double> p(df.size());
  for (int i=0; i<df.size(); i++)
    p[i] = boost::math::pdf(boost::math::students_t(df.at(i)), x.at(i)) / scale.at(i);

  return xt::adapt(p);
}

void StudentT::update_theta(double data) {
  auto mu0 = xt::concatenate(xt::xtuple(m_mu0, (m_kappa*m_mu + data)/(m_kappa + 1.0)));
  auto kappa0 = xt::concatenate(xt::xtuple(m_kappa0, m_kappa+1.0));
  auto alpha0 = xt::concatenate(xt::xtuple(m_alpha0, m_alpha+0.5));
  m_beta = xt::concatenate(xt::xtuple(m_beta0, m_beta+(m_kappa*xt::pow((data-m_mu),2))/(2.0*(m_kappa+1.0))));
  m_mu = mu0;
  m_kappa = kappa0;
  m_alpha = alpha0;
}

BOCD::BOCD(int lam, double alpha, double beta, double kappa, double mu) :
  m_hazard_function(lam), m_observation_likelihood(alpha, beta, kappa, mu), m_t0(0), m_t(-1),
  m_growth_probs({1.0}) {
}

void BOCD::update(double data) {
  m_t += 1;

  size_t t = m_t - m_t0;

  if(m_growth_probs.size() == t+1) {
    auto temp = m_growth_probs;
    m_growth_probs.resize({(t+1)*2});
    xt::view(m_growth_probs, xt::range(0, t+1)) = temp;
    xt::view(m_growth_probs, xt::range(t+1, (t+1)*2)) = temp;
  }

  auto pred_probs = m_observation_likelihood.pdf(data);

  auto H = m_hazard_function(t+1);
  
  double cp_prob = xt::sum(xt::view(m_growth_probs, xt::range(0, t+1)) * pred_probs * H)(0);

  xt::view(m_growth_probs, xt::range(1, t+2)) = xt::view(m_growth_probs, xt::range(0, t+1)) * pred_probs * (1-H);

  m_growth_probs.data()[0] = cp_prob;

  xt::view(m_growth_probs, xt::range(0, t+2)) = xt::view(m_growth_probs, xt::range(0, t+2)) 
     / xt::sum(xt::view(m_growth_probs, xt::range(0, t+2)));
  
  m_observation_likelihood.update_theta(data);
}
