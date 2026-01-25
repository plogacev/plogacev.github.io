functions {

  array[] vector local_weighted_histograms(int t_mid, int window_size, array[,] int histogram, vector log_ncp_probs)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      
      int window_size_left  = min(window_size, t_mid-1);
      int window_size_right = min(window_size, n_time_points-t_mid);
//print("window_size_left: ", window_size_left);
//print("window_size_right: ", window_size_right);

      real epsilon = 1e-6;
      vector[n_price_points] complete_segment_histogram = to_vector(histogram[t_mid]) + epsilon;
      vector[n_price_points] left_segment_histogram = complete_segment_histogram; // = to_vector(histogram[t_mid]);
      vector[n_price_points] right_segment_histogram; //= to_vector(histogram[t_mid+1]) + epsilon;

      if (window_size_left > 0) {
          real log_weight = 0.0;
          for (s in 1:window_size_left ) {
              int t_cur = t_mid - s;
              //print("t_cur: ", t_cur);
              log_weight += log_ncp_probs[t_cur];
              complete_segment_histogram += to_vector(histogram[t_cur]) * exp(log_weight);
              if ( s <= window_size-1 ) {
                  left_segment_histogram = complete_segment_histogram;
              }
          }
      }
      //print("complete_segment_histogram: ", complete_segment_histogram);
      //print("left_segment_histogram: ", left_segment_histogram);

      if (window_size_right > 0) {
          real log_weight = 0.0;
          for (s in 1:window_size_right) {
              int t_cur = t_mid + s;
              //print("t_cur: ", t_cur);
              log_weight += log_ncp_probs[t_cur - 1];
              complete_segment_histogram += to_vector(histogram[t_cur]) * exp(log_weight);
          }

          right_segment_histogram = rep_vector( epsilon, n_price_points );
          log_weight = 0.0;
          for (s in 1:window_size_right) {
              int t_cur = t_mid + s;
              right_segment_histogram += to_vector(histogram[t_cur]) * exp(log_weight);
              if ( s < window_size_right ) {
                  log_weight += log_ncp_probs[t_cur];
              }
          }
      }

      array[3] vector[n_price_points] results;
      results[1] = complete_segment_histogram;
      results[2] = left_segment_histogram;
      results[3] = right_segment_histogram;

      return results;
  }

  real delta_histogram_mean( vector left_segment_histogram, vector right_segment_histogram, real ncp_prob, vector price_points) {
      real mean_left = dot_product(left_segment_histogram, price_points) / sum(left_segment_histogram);
      real mean_right = dot_product(right_segment_histogram, price_points) / sum(right_segment_histogram);
      return abs(mean_right-mean_left);
  }

}

data {
  int<lower=2> n_time_points;
  int<lower=1> n_price_points;
  array[n_time_points, n_price_points] int histogram;
  vector[n_price_points] price_points;
  int<lower=1> window_size;
  
  //vector<upper=0>[n_time_points-1] test_log_ncp_probs;

}

parameters {
    real<lower=0> ncp_sigma;
    real<lower=0.1> cp_mode;
    real<lower=0> cp_scale;
    real<lower=0, upper=1> cp_theta;

    //vector<lower=0.0001, upper=0.9999>[n_time_points-1] cp_probs;
    vector<upper=0>[n_time_points-1] log_ncp_probs;
}
transformed parameters {
    real cp_alpha = 1 + 1 / cp_scale;
    real cp_beta = 1 / (cp_scale * cp_mode);
}
model {

  //log_ncp_probs ~ normal(5, 10);
  cp_mode ~ normal(1, 5);
  cp_scale ~ exponential(1);
  ncp_sigma ~ exponential(1);
  cp_theta ~ beta(1, 5);

  for (t in 1:n_time_points) {
      array[3] vector[n_price_points] histograms = local_weighted_histograms(t, window_size, histogram, log_ncp_probs);
      vector[n_price_points] complete_segment_histogram = histograms[1]; 

      vector[n_price_points] local_tau = complete_segment_histogram / sum(complete_segment_histogram);

      if ( sum(histogram[t]) > 0 ) {
        histogram[t] ~ multinomial(local_tau);
      }
      
      if (t < n_time_points) {
          // to-do: the differences should probably be modeled as latent variables, but that requires to model the entire distribution over time.
          vector[n_price_points] left_segment_histogram = histograms[2];
          vector[n_price_points] right_segment_histogram = histograms[3];
          real delta = delta_histogram_mean( left_segment_histogram, right_segment_histogram, exp(log_ncp_probs[t]), price_points);

          //real log_lik_delta_prior_ncp = normal_lpdf(delta | 0, ncp_sigma);
          //real log_lik_delta_prior_cp = normal_lpdf(delta | cp_mean, cp_sigma);
          real log_lik_delta_prior_ncp = normal_lpdf(delta | 0, ncp_sigma) + log(2);  // Half-normal
          real log_lik_delta_prior_cp = gamma_lpdf(delta | cp_alpha, cp_beta);                     // Gamma(shape=4, rate=2)
          real log_lik_delta_prior = log_mix(cp_theta, log_lik_delta_prior_cp, log_lik_delta_prior_ncp);
          
          target += log_lik_delta_prior;
      }
  }
}

generated quantities {
  vector[n_time_points-1] deltas;
  
  for (t in 1:(n_time_points-1)) {
      array[3] vector[n_price_points] histograms = local_weighted_histograms(t, window_size, histogram, log_ncp_probs);

      vector[n_price_points] left_segment_histogram = histograms[2] / sum(histograms[2]);
      vector[n_price_points] right_segment_histogram = histograms[3] / sum(histograms[3]);
      real delta = delta_histogram_mean( left_segment_histogram, right_segment_histogram, exp(log_ncp_probs[t]), price_points);
      deltas[t] = delta;
  }

}
