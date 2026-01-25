functions {
  
  
  // Local smoothed histograms (left), weighted by changepoint probabilities.
  // Provides continuous approximations to left right segments around a potential changepoint.
  vector local_weighted_histogram_left(int t_start, int window_size, array[,] int histogram, vector log_ncp_probs)
  {
      int n_price_points = dims(histogram)[2];

      real epsilon = 1e-10;
      vector[n_price_points] histogram_segment = to_vector(histogram[t_start]) + rep_vector( epsilon, n_price_points );

      if ( (window_size-1) > 0) {
          real log_weight = 0.0;
          for ( s in 1:(window_size-1) ) {
              int t_cur = t_start - s;
              log_weight += log_ncp_probs[t_cur];
              histogram_segment += to_vector(histogram[t_cur]) * exp(log_weight);
          }
      }
      
      return histogram_segment;
  }


  // Local smoothed histograms (right), weighted by changepoint probabilities.
  // Provides continuous approximations to left right segments around a potential changepoint.
  vector local_weighted_histogram_right(int t_start, int window_size, array[,] int histogram, vector log_ncp_probs)
  {
      int n_price_points = dims(histogram)[2];

      real epsilon = 1e-10;
      vector[n_price_points] histogram_segment = to_vector(histogram[t_start]) + rep_vector( epsilon, n_price_points );
      
      if ( (window_size-1) > 0) {
          real log_weight = 0.0;
          for ( s in 1:(window_size-1) ) {
              int t_cur = t_start + s;
              log_weight += log_ncp_probs[t_cur-1];
              histogram_segment += to_vector(histogram[t_cur]) * exp(log_weight);
          }
      }
      
      return histogram_segment;
  }

  real compute_change_magnitudes_mean(vector left_histogram, vector right_histogram, vector price_points) {
      real right_mean = dot_product(right_histogram, price_points) / sum(right_histogram);
      real left_mean = dot_product(left_histogram, price_points) / sum(left_histogram);
      return (right_mean - left_mean);
  }

  real compute_change_magnitudes_emd(vector h_left, vector h_right, vector price_points) {
      vector[rows(h_left)] cdf_left = cumulative_sum(h_left) / sum(h_left);
      vector[rows(h_right)] cdf_right = cumulative_sum(h_right) / sum(h_right);
  
      // Distance over the price axis
      return sum(fabs(cdf_right - cdf_left) .* price_points);
  }

  // For each time point, calculates a proxy for the magnitude of regime change (currently via mean price difference) between adjacent windows.
  // to-do: think of a more robust metric than the mean, something distributional
  // to-do: apply shrinking through a prior, taking into account the sample sizes
  vector compute_change_magnitudes(array[,] int histogram, vector price_points, int window_size, vector lp_ncp)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      vector[n_time_points-1] deltas;
      
      // Compare each time step with previous
      // to-do: Compute a weighted local window like in the window algorithm
      for (t in 1:(n_time_points-1))
      {
          int window_size_left  = min(window_size, t);
          int window_size_right = min(window_size, n_time_points-t);
    
          vector[n_price_points] left_segment_histogram = local_weighted_histogram_left(t, window_size_left, histogram, lp_ncp);
          vector[n_price_points] right_segment_histogram = local_weighted_histogram_right(t+1, window_size_right, histogram, lp_ncp);

          //deltas[t] = compute_change_magnitude(left_segment_histogram, right_segment_histogram, price_points);
          deltas[t] = compute_change_magnitudes_mean(left_segment_histogram, right_segment_histogram, price_points);
      }
      
      return deltas;
  }
  
}


data {
  int<lower=1> n_time_points;
  int<lower=1> n_price_points;
  array[n_time_points, n_price_points] int histogram;
  vector[n_price_points] price_points;
  
  int change_window_size;
  real prior_cp_probs_one;
  real prior_change_magnitude_min;
  real prior_change_magnitude_typical;
}

parameters {
  real<lower=0> prior_change_cp_mu;
  real<lower=0> prior_change_cp_sigma;
  real<lower=0> prior_change_ncp_sigma;

  vector<upper=0>[n_time_points-1] lp_cp;
  real<lower=2> mean_segment_duration;


}

transformed parameters {
  vector<upper=0>[n_time_points-1] lp_ncp = log1m_exp(lp_cp);
  
  vector[n_time_points-1] change_magnitudes = compute_change_magnitudes(histogram, price_points, change_window_size, lp_ncp);
}

model {
   prior_change_cp_mu ~ normal(2, 1);
   prior_change_cp_sigma ~ exponential(1);
   prior_change_ncp_sigma ~ exponential(1);

  mean_segment_duration ~ normal(10*250, 100);
  real lperc_cp_one = log(1/mean_segment_duration);

  // to-do: use the kind of prior that sets most values to almost exactly 0
  real epsilon = 1e-12;
  for (i in 1:(n_time_points-1)) {
    real prior_cp_magnitude = normal_lpdf( abs(change_magnitudes[i]) + epsilon | prior_change_cp_mu, prior_change_cp_sigma );
    real prior_ncp_magnitude = normal_lpdf( abs(change_magnitudes[i]) + epsilon | 0, prior_change_ncp_sigma );
    target += log_sum_exp( lperc_cp_one + lp_cp[i] + prior_cp_magnitude, log1m_exp(lperc_cp_one) + lp_ncp[i] + prior_ncp_magnitude );
  }

  //target += compute_marginal_loglik(lp_cp, lp_ncp, segments_loglik); //, lp_cp_magnitude_prior, lp_ncp_magnitude_prior
}
