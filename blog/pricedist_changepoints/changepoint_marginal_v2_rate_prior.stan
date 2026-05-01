functions {
  
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

  real compute_marginal_loglik_path(int t1, int t2, int n_time_points, 
                                    vector lp_cp, vector lp_ncp) // , vector lp_cp_magnitude_prior, vector lp_ncp_magnitude_prior
  {
      real lp_path = 0.0;
      //real lp_segment_change = 0.0;

      if ( t2 < n_time_points ) { // terminates with a changepoint at the end of the segment
          lp_path += lp_cp[t2];
          //lp_segment_change += lp_cp_magnitude_prior[t2];
      }
      if ( t1 < t2 ) {  // aggregate all the non-changes before the changepoint 
          lp_path += sum( lp_ncp[t1:(t2 - 1)] );
          //lp_segment_change += sum( lp_ncp_magnitude_prior[t1:(t2 - 1)] );
      }

      // Total log-prob for this segmentation path
      return lp_path;// + lp_segment_change;
  }

  // Main forward pass algorithm: computes marginal log-likelihood via dynamic programming
  // Each time step t2 accumulates total log-probabilities over all segmentations ending at t2
  real compute_marginal_loglik(vector lp_cp, vector lp_ncp, matrix segments_loglik) // , vector lp_cp_magnitude_prior, vector lp_ncp_magnitude_prior
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
                real cur_segment_loglik = compute_marginal_loglik_path(t1, t2, n_time_points, lp_cp, lp_ncp); // , lp_cp_magnitude_prior, lp_ncp_magnitude_prior

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
  real<lower=2> mean_segment_duration;
  vector<upper=0>[n_time_points-1] lp_cp;
}

transformed parameters {
  vector<upper=0>[n_time_points-1] lp_ncp = log1m_exp(lp_cp);
}

model {
  
  mean_segment_duration ~ normal(250, 100);
  real lperc_cp_one = log(1/mean_segment_duration);
  for (i in 1:(n_time_points-1)) {
    target += log_sum_exp( lperc_cp_one + lp_cp[i], log1m_exp(lperc_cp_one) + lp_ncp[i] );
  }

  target += compute_marginal_loglik(lp_cp, lp_ncp, segments_loglik); //, lp_cp_magnitude_prior, lp_ncp_magnitude_prior
}
