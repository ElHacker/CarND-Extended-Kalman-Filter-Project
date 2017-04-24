#include "kalman_filter.h"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // x' = F * x + noise
  // Note: noise is always 0, might as well not use it in the equation.
  // noise = VectorXd(2);
  // noise << 0, 0;
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // new state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools;
  MatrixXd jacobian = tools.CalculateJacobian(x_);
  VectorXd y = z - MapCartesianToPolar(x_);
  MatrixXd jacobianT = jacobian.transpose();
  MatrixXd S = jacobian * P_ * jacobianT + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * jacobianT * Si;

  // new state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::MapCartesianToPolar(VectorXd& x_state) {
  VectorXd result(3);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float range = sqrt(pow(px, 2) + pow(py, 2));
  float bearing = atan(py / px);
  float rangeRate = (px * vx + py * vy) / range;

  result << range,
         bearing,
         rangeRate;
  return result;
}
