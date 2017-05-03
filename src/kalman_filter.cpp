#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using namespace std;
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
  MatrixXd K = (P_ * Ht) * Si;

  // new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = z - MapCartesianToPolar(x_);
  // Adjust angle phi (bearing) to be between -PI and PI;
  while (y[1] < -M_PI) {
     y[1] += 2 * M_PI;
  }
  while (y[1] > M_PI) {
    y[1] -= 2 * M_PI;
  }
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = (P_ * Ht) * Si;

  // new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::MapCartesianToPolar(VectorXd& x_state) {
  VectorXd result(3);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  // rho
  float range = sqrt(pow(px, 2) + pow(py, 2));
  // avoid division by zero
  if(fabs(px) < 0.0001){
    cout << "MapCartesianToPolar() - Error: px Division by Zero" << endl;
  }
  // phi
  float bearing = atan2(py, px);
  if (fabs(range) < 0.0001) {
    cout << "MapCartesianToPolar() - Error: range Division by Zero" << endl;
    return result;
  }
  // rhodot
  float rangeRate = (px * vx + py * vy) / range;

  result << range,
         bearing,
         rangeRate;
  return result;
}
