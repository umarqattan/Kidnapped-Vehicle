/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    
    default_random_engine generator;
    normal_distribution<double> x_noise(x, std[0]);
    normal_distribution<double> y_noise(y, std[1]);
    normal_distribution<double> theta_noise(theta, std[2]);
    
    for (unsigned i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x_noise(generator);
        p.y = y_noise(generator);
        p.theta = theta_noise(generator);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine generator;
    normal_distribution<double> x_noise(0, std_pos[0]);
    normal_distribution<double> y_noise(0, std_pos[1]);
    normal_distribution<double> theta_noise(0, std_pos[2]);
    
    for (unsigned i = 0; i < num_particles; i++) {
        if (yaw_rate == 0) {
            particles[i].x += x_noise(generator) + velocity*cos(particles[i].theta)*delta_t;
            particles[i].y += y_noise(generator) + velocity*sin(particles[i].theta)*delta_t;
        } else {
            particles[i].x += x_noise(generator) + (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            particles[i].y += y_noise(generator) + (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            particles[i].theta += theta_noise(generator) + yaw_rate*delta_t;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (unsigned i = 0; i < observations.size(); i++ ) {
        LandmarkObs& obs = observations[i];
        double closest_distance = numeric_limits<double>::max();
        for (unsigned j = 0; j < observations.size(); j++) {
            LandmarkObs pred = predicted[j];
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < closest_distance) {
                closest_distance = distance;
                obs.id = j;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    for (unsigned i = 0; i < num_particles; i++) {
        Particle& particle = particles[i];
        
        vector<LandmarkObs> transformed_observations;
        
        for (unsigned j = 0; j < observations.size(); j++) {
            LandmarkObs obs = observations[j];
            LandmarkObs transformed_obs;
            transformed_obs.x = obs.x*cos(particle.theta) - obs.y*sin(particle.theta) + particle.x;
            transformed_obs.y = obs.x*sin(particle.theta) + obs.y*cos(particle.theta)+particle.y;
            
            transformed_observations.push_back(transformed_obs);
        }
        
        vector<LandmarkObs> predictions;
        for (unsigned l = 0; l < map_landmarks.landmark_list.size(); l++) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[l];
            double distance_from_landmark = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
            if (distance_from_landmark < sensor_range) {
                LandmarkObs pred = {landmark.id_i, landmark.x_f, landmark.y_f};
                predictions.push_back(pred);
            }
        }
        dataAssociation(predictions, transformed_observations);
        
    
        particle.weight = 1.0;
        for (int k = 0; k < transformed_observations.size(); k++) {
            LandmarkObs obs = transformed_observations[k];
            LandmarkObs pred = predictions[obs.id];
            
            // mult-variate Gaussian distribution
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double normalizer = 1/(2*M_PI*std_x*std_y);
            double dx = obs.x-pred.x;
            double dy = obs.y-pred.y;
            double dx_2 = dx*dx;
            double dy_2 = dy*dy;
            double probability = normalizer*exp(-((dx_2/2 * std_x*std_x) + (dy_2/2 * std_y*std_y)));
            
            particle.weight = particle.weight * probability;
        }

        weights[i] = particle.weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine generator;
    discrete_distribution<> weight_distribution(weights.begin(), weights.end());
    vector<Particle> resampled_particles;
    
    for (unsigned i = 0; i < num_particles; i++) {
        Particle resampled_particle = particles[weight_distribution(generator)];
        
        resampled_particles.push_back(resampled_particle);
    }
    
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
