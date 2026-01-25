functions {
  
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
  
  // Local smoothed histograms (left and right), weighted by changepoint probabilities.
  // Provides continuous approximations to left right segments around a potential changepoint.
  array[] vector local_weighted_histograms(int t_mid, int window_size, array[,] int histogram, vector log_ncp_probs)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      
      int window_size_left  = min(window_size, t_mid);
      int window_size_right = min(window_size, n_time_points-t_mid);

      //print("t ", t_mid, " [", t_mid-window_size_left, "; ", t_mid, "], [", t_mid+1, "; ", t_mid+1+window_size_right, "]");
      //print("  ", t_mid, " (", t_mid, ", ", window_size_left, "), (", t_mid+1, ", ", window_size_right, ")");

      real epsilon = 1e-10;
      vector[n_price_points] left_segment_histogram = local_weighted_histogram_left(t_mid, window_size_left, histogram, log_ncp_probs);
      //print("-");
      vector[n_price_points] right_segment_histogram = local_weighted_histogram_right(t_mid+1, window_size_right, histogram, log_ncp_probs);

      //print("/t");

      array[2] vector[n_price_points] results;
      results[1] = left_segment_histogram;
      results[2] = right_segment_histogram;

      return results;
  }

  // Computes row-wise cumulative sum of histograms for fast subsegment histogram extraction.
  array[,] int create_histogram_cumulative(array[,] int histogram)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      array[n_time_points, n_price_points] int result;
      
      // Initialize with first row
      result[1] = histogram[1];

      // Reject if all values are zero in the entire histogram
      if( sum(histogram[1]) == 0 ) {
          reject("Row 1 in quantity histogram is all zeroes. Please exclude such rows.");
      }
      
      // Compute cumulative sum row-wise
      for (t in 2:n_time_points) {
          for (p in 1:n_price_points)
            result[t][p] = result[t - 1][p] + histogram[t][p];
          
          // Check for empty histogram rows (no prices at all)
          if ( sum(histogram[t]) == 0 ) {
            reject("Row ", t, " in quantity histogram is all zeroes. Please exclude such rows.");
          }

      }
      return result;
  }

  // Retrieves total histogram for a given [t_start, t_end] segment using cumulative histograms.
  vector extract_segment_histogram(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      array[n_price_points] int start_vals;
      array[n_price_points] int end_vals;
      array[n_price_points] int result;
      
      
      // Retrieve cumulative before the start
      int t_prev = t_start - 1;
      if (t_prev < 1) {
          for (p in 1:n_price_points)
              start_vals[p] = 0;
      } else {
          start_vals = histogram_cumulative[t_prev];
      }

      // Compute end cumulative value
      end_vals = (t_end <= n_time_points) ? histogram_cumulative[t_end] :  histogram_cumulative[n_time_points];
      
      // Segment = difference between cumulative ends
      for (p in 1:n_price_points)
          result[p] = end_vals[p] - start_vals[p];

      return to_vector(result);
  }

  // Computes self-likelihood of each segment under a multinomial model, assuming its empirical histogram's proportions are the MLE.
  real compute_segment_loglik(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_price_points = dims(histogram_cumulative)[2];

      real epsilon = 1e-10;
      vector[n_price_points] histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative) + epsilon;
      vector[n_price_points] log_probs = log( histogram / sum(histogram) ); 

      // Return segment log-likelihood
      return dot_product(histogram, log_probs);
  }

  // Precomputes the matrix of log-likelihoods for all possible segments.
  matrix compute_segments_loglik(array[,] int histogram_cumulative)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      matrix[n_time_points, n_time_points] segments_loglik;
  
      // Compute upper triangle log-likelihoods
      for (t1 in 1:n_time_points) {
          for (t2 in t1:n_time_points) {
              segments_loglik[t1, t2] = compute_segment_loglik(t1, t2, histogram_cumulative);
          }
      }

      // Set lower triangle to -Inf (not valid segments)
      for (t1 in 1:n_time_points) {
          for (t2 in 1:(t1 - 1)) {
              segments_loglik[t1, t2] = negative_infinity();
          }
      }
  
      return segments_loglik;
  }

  // Applied at each timepoint, calculates a proxy for the magnitude of regime change (currently via mean price difference)
  // between adjacent windows.
  real compute_change_magnitude(vector histogram_prev, vector histogram_curr, vector price_points)
  {
      // Retrieve previous and current segment
      real mean_prev = dot_product(histogram_prev, price_points) / sum(histogram_prev);
      real mean_curr = dot_product(histogram_curr, price_points) / sum(histogram_curr);
      
      // Change magnitude is absolute mean difference
      return (mean_prev - mean_curr);
  }

  // For each time point, calculates a proxy for the magnitude of regime change (currently via mean price difference) between adjacent windows.
  // to-do: think of a more robust metric than the mean, something distributional
  // to-do: apply shrinking through a prior, taking into account the sample sizes
  vector compute_change_magnitudes(array[,] int histogram, array[,] int histogram_cumulative, vector price_points, int window_size, vector lp_ncp)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      vector[n_time_points-1] deltas;
      
      // Compare each time step with previous
      // to-do: Compute a weighted local window like in the window algorithm
      for (t in 1:(n_time_points-1)) {
          array[2] vector[n_price_points] local_hist = local_weighted_histograms(t, window_size, histogram, lp_ncp);
          vector[n_price_points] left_segment_histogram = local_hist[1];
          vector[n_price_points] right_segment_histogram = local_hist[2];

          //deltas[t] = compute_change_magnitude(left_segment_histogram, right_segment_histogram, price_points);
          deltas[t] = dot_product(right_segment_histogram, price_points) / sum(right_segment_histogram) - dot_product(left_segment_histogram, price_points) / sum(left_segment_histogram);

      }
      
      return deltas;
  }

  // to-do: review this prior; it's shape is more that of a CDF
  real changepoint_magnitude_prior(real x, real percentile_5, real percentile_50) {
    real k = log(19.0) / (percentile_50 - percentile_5); 
    return -log1p_exp(-k * (x - percentile_50));
  }

  // Applies change magnitude prior to the change magnitude
  // These priors serve as a penalty or encouragement for placing changepoints
  vector compute_lp_change_prior(vector change_magnitude, vector lp_cp, real prior_alpha, real prior_sigma)
  {
      int T = num_elements(change_magnitude);
      vector[T] lp;

      for (t in 1:T) {
          lp[t] = changepoint_magnitude_prior(change_magnitude[t], 0.01, 0.1);
      }

      return lp;
  }

  real compute_marginal_loglik_path(int t1, int t2, int n_time_points, vector lp_cp, vector lp_ncp, vector lp_cp_magnitude_prior, vector lp_ncp_magnitude_prior)
  {
      real lp_path = 0.0;
      real lp_segment_change = 0.0;

      if ( t2 < n_time_points ) { // terminates with a changepoint at the end of the segment
          lp_path += lp_cp[t2];
          lp_segment_change += lp_cp_magnitude_prior[t2];
      }
      if ( t1 < t2 ) {  // aggregate all the non-changes before the changepoint 
          lp_path += sum( lp_ncp[t1:(t2 - 1)] );
          lp_segment_change += sum( lp_ncp_magnitude_prior[t1:(t2 - 1)] );
      }

      // Total log-prob for this segmentation path
      return lp_path;// + lp_segment_change;
  }

  // Main forward pass algorithm: computes marginal log-likelihood via dynamic programming
  // Each time step t2 accumulates total log-probabilities over all segmentations ending at t2
  real compute_marginal_loglik(vector lp_cp, vector lp_ncp, matrix segments_loglik, vector lp_cp_magnitude_prior, vector lp_ncp_magnitude_prior)
  {
        int n_time_points = dims(segments_loglik)[1];
        int n_cp = n_time_points - 1;

        vector[n_time_points] marginal_loglik;

        // Dynamic programming over segment ends
        for (t2 in 1:n_time_points) {
            vector[t2] path_lls;
    
            // Iterate over all previous segmentations ending at t1
            for (t1 in 1:t2) {
              
                //  retrieve cumulative log-prob up to previous segment
                real prev_marginal_loglik = (t1 > 1) ? marginal_loglik[t1-1] : 0;

                // compute the log-prob for the presently considered segment
                real cur_segment_loglik = compute_marginal_loglik_path(t1, t2, n_time_points, lp_cp, lp_ncp, lp_cp_magnitude_prior, lp_ncp_magnitude_prior);

                // log-prob for the presently considered segment + the marginal for the paths that may lead to it
                real cur_path_loglik = cur_segment_loglik + prev_marginal_loglik;
                
                // log-prob for the emission
                real segment_loglik = segments_loglik[t1, t2];

                // log-prob for the current path + emission 
                path_lls[t1] = cur_path_loglik + segment_loglik;
            }
    
            // aggregate over all paths leading up to and including t2
            marginal_loglik[t2] = log_sum_exp(path_lls);
        }
    
        // return the log-likelihood for all the paths leading to the end point
        return marginal_loglik[n_time_points];
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

transformed data {
  array[n_time_points, n_price_points] int histogram_cumulative = create_histogram_cumulative(histogram);
  matrix[n_time_points, n_time_points] segments_loglik = compute_segments_loglik(histogram_cumulative);
}

parameters {
  real<lower=1> lambda;
  
  real<lower=0> prior_change_ncp_sigma;
  real<lower=0.05> prior_change_cp_mu;
  real<lower=0> prior_change_cp_sigma;

  vector<upper=0>[n_time_points-1] lp_cp;
  
}

transformed parameters {
/*
  vector<upper=0>[n_time_points-1] lp_cp = rep_vector(-10, n_time_points-1);
  {
    // 98, 
    //array[10] int indices = {100, 201, 207, 302, 398, 404, 496, 545, 608, 609};
    array[1] int indices = {24};
    real alomost_one = 0; //-0.00001;
    for (idx in indices) {
        lp_cp[idx] = alomost_one;
    }
  }
*/

  vector<upper=0>[n_time_points-1] lp_ncp = log1m_exp(lp_cp);
  vector[n_time_points-1] change_magnitudes = compute_change_magnitudes(histogram, histogram_cumulative, price_points, change_window_size, lp_ncp);
  //vector[n_time_points-1] lp_change_prior = rep_vector(0.0, n_time_points-1); //compute_lp_change_prior(change_magnitudes, lp_cp, prior_alpha, prior_sigma);

/*
  vector[n_time_points-1] left_avg = left_compute_change_magnitudes_(histogram, histogram_cumulative, price_points, change_window_size, lp_ncp);
  vector[n_time_points-1] right_avg = right_compute_change_magnitudes_(histogram, histogram_cumulative, price_points, change_window_size, lp_ncp);
  vector[n_time_points-1] change_magnitudes_alt = right_avg - left_avg;
*/

  vector[n_time_points-1] lp_cp_magnitude_prior;
  vector[n_time_points-1] lp_ncp_magnitude_prior;
  for ( t in 1:(n_time_points-1) ) {
      lp_cp_magnitude_prior[t] = normal_lpdf( abs(change_magnitudes[t]) | 0.5, prior_change_cp_sigma  );
      lp_ncp_magnitude_prior[t] = normal_lpdf( abs(change_magnitudes[t]) | 0, prior_change_ncp_sigma );
  }
}

model {

  prior_change_ncp_sigma ~ exponential(1);
  prior_change_cp_sigma ~ exponential(1);
  prior_change_cp_mu ~ normal(1, 0.5);

  //lp_cp ~ normal(-50, 10);


  lambda ~ normal(5000, 100);
  //real lperc_cp_one = log(1/lambda);
  real lperc_cp_one = log(1/1);
  
  for (i in 1:(n_time_points-1)) {
    target += log_sum_exp( lperc_cp_one + lp_cp[i], log1m_exp(lperc_cp_one) + lp_ncp[i] );
  }

  target += compute_marginal_loglik(lp_cp, lp_ncp, segments_loglik, lp_cp_magnitude_prior, lp_ncp_magnitude_prior);
}
