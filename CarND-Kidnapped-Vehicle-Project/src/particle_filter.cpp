/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"


using namespace std;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
 
  
  num_particles = 1000;
  
  
  double std_x, std_y, std_theta;
  default_random_engine gen;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  
  normal_distribution<double> dist_x(x,std_x);
  normal_distribution<double> dist_y(y,std_y);
  normal_distribution<double> dist_theta(theta,std_theta);
  
  weights.resize(num_particles);
  particles.resize(num_particles);
  
   //initializing 
  for(int i = 0; i < num_particles; ++i){
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0; 
    particles[i] = p;
  }
  is_initialized = true;
}
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;

   //generating normal distribution
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  //creating new state
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    //adding noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
  //dataassociation
   for(unsigned int i = 0; i < observations.size(); ++i) {
   
    double min_dist = INFINITY;
    int closest_particle_id = -1;

    for(unsigned int j = 0; j < predicted.size(); ++j) {
      
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (distance < min_dist) {
        min_dist = distance;
        closest_particle_id = j;
      }
    }
    observations[i].id = closest_particle_id;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  //Calculations
 
  unsigned int landmark_size = map_landmarks.landmark_list.size();
  
  auto normalizer = 1 / (2*M_PI*std_landmark[0]*std_landmark[1]);
  auto sigma_x = 2*pow(std_landmark[0],2);
  auto sigma_y = 2*pow(std_landmark[1],2);
  
  for(int i = 0; i < num_particles; ++i){
     double x_p = particles[i].x;
     double y_p = particles[i].y;
     double theta_p = particles[i].theta;
    
   // choosing landmarks within sensor_range 
    vector<LandmarkObs> landmark_p;
    for(unsigned int j = 0; j < landmark_size; ++j){
      auto x_l = map_landmarks.landmark_list[j].x_f;
      auto y_l = map_landmarks.landmark_list[j].y_f;
      auto id_l = map_landmarks.landmark_list[j].id_i;
      
      if(dist(x_l,y_l,x_p,y_p) <= sensor_range){
        landmark_p.push_back(LandmarkObs{id_l,x_l,y_l});
        }
      }
    
  //setting weights
    // transforming
    if(landmark_p.size() == 0){
      particles[i].weight = 0;
      weights[i] = 0;
    }
    else{
      vector<LandmarkObs> obs_transformed;
      for(unsigned int t = 0; t < observations.size(); ++t){
        auto x_map = (cos(theta_p) * observations[t].x) - (sin(theta_p) * observations[t].y) + x_p;
        auto y_map =(sin(theta_p) * observations[t].x) + (cos(theta_p) * observations[t].y) + y_p;
        obs_transformed.push_back(LandmarkObs{observations[t].id,x_map,y_map});
      }
      
    // associating
      dataAssociation(landmark_p,obs_transformed);
      
    //calculating weight
     auto total_prob = 1.0;
     for(unsigned int p = 0 ; p < obs_transformed.size(); ++p){
       auto o = obs_transformed[p];
       auto pr = landmark_p[o.id];
       auto d_x = o.x - pr.x;
       auto d_y = o.y - pr.y;
       auto exponent = (pow(d_x,2) / sigma_x) + (pow(d_y,2) / sigma_y);
       
       total_prob *= exp(-exponent) * normalizer ;
     }
     particles[i].weight = total_prob;
      weights[i] = total_prob;
    }
  }
}


void ParticleFilter::resample() {
  /**
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  //resampling
  discrete_distribution<int> dist_resample(weights.begin(), weights.end());
  default_random_engine gen;
  
  vector<Particle> resampled_particles(particles.size());
  
  for(unsigned int i = 0 ; i < particles.size(); ++i){
    resampled_particles[i] = particles[dist_resample(gen)];
  }
  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}