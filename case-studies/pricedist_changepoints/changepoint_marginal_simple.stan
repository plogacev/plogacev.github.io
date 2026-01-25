functions {
  
  // Computes cumulative sum of the histogram over time
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

  // Extracts the histogram for a segment [t_start, t_end]
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

  // Computes segment log-likelihood under multinomial model
  real compute_segment_loglik(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_price_points = dims(histogram_cumulative)[2];
      vector[n_price_points] histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative);
      real total_qty = sum(histogram);
      vector[n_price_points] log_probs;
  
      // Compute log-probabilities for multinomial
      for (i in 1:n_price_points) {
          log_probs[i] = histogram[i] > 0 ? log(histogram[i] / total_qty) : 0.0;
      }

      // Return segment log-likelihood
      return total_qty > 0 ? dot_product(histogram, log_probs) : 0.0;
  }

  // Precomputes all segment log-likelihoods for dynamic programming algo
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

  // Computes segment-by-segment difference statistic (e.g., mean price here) as a proxy for change magnitude
  // to-do: implement it by segment (with reasonable pruning)
  real compute_change_magnitude(vector histogram_prev, vector histogram_curr, vector price_points)
  {
      // Retrieve previous and current segment
      real mean_prev = dot_product(histogram_prev, price_points) / sum(histogram_prev);
      real mean_curr = dot_product(histogram_curr, price_points) / sum(histogram_curr);
      
      // Change magnitude is absolute mean difference
      return abs(mean_prev - mean_curr);
  }

  // Computes change magnitude between adjacent time steps
  // to-do: think of a more robust metric than the mean, something distributional
  // to-do: apply shrinking through a prior, taking into account the sample sizes
  vector compute_change_magnitudes(array[,] int histogram_cumulative, vector price_points)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      vector[n_time_points] deltas;
      
       // No change defined before time 1
       deltas[1] = 0.0;
      
      // Compare each time step with previous
      // to-do: Compute a weighted local window like in the window algorithm
      for (t in 2:n_time_points) {
          vector[n_price_points] histogram_prev = extract_segment_histogram(t - 1, t - 1, histogram_cumulative);
          vector[n_price_points] histogram_curr = extract_segment_histogram(t,     t,    histogram_cumulative);
          deltas[t] = compute_change_magnitude(histogram_prev, histogram_curr, price_points);
      }
  
      return deltas;
  }

  // Applies change magnitude prior to the change magnitude
  // These priors serve as a penalty or encouragement for placing changepoints
  vector compute_lp_change_prior(vector change_magnitude, real prior_mu, real prior_sigma)
  {
      int T = num_elements(change_magnitude);
      vector[T] lp;

      for (t in 1:T) {
        // to-do
          // to-do: [BUG] Remove or figure out why this prior works in very odd ways: When I set 
          // all change_magnitude[t] values to 0, the optim results stay nearly the same. When
          // I set all lp_change_prior values below  to 0, there are more changepoints.
          // Unsurprisingly, widening the window doesn't change things even one bit.
          lp[t] = normal_lpdf( change_magnitude[t] | prior_mu, prior_sigma);
      }
  
      return lp;
  }

  // Main forward pass algorithm: computes marginal log-likelihood via dynamic programming
  // Each time step t2 accumulates total log-probabilities over all segmentations ending at t2
  real compute_marginal_loglik(vector cp_probs_raw, matrix segments_loglik, vector lp_change_prior)
  {
        int n_time_points = dims(segments_loglik)[1];
        int n_cp = n_time_points - 1;
        real epsilon = 1e-12;
        
        
        // Climp cp_probs to avoid numerical errors
        vector[n_cp] cp_probs = fmin(fmax(cp_probs_raw, epsilon), 1 - epsilon);
        vector[n_cp] lp_cp = log(cp_probs);
        vector[n_cp] lp1m_cp = log1m(cp_probs);
        vector[n_time_points + 1] marginal_loglik;
        //vector[n_time_points + 1] marginal_loglik;
    
        // base case: empty segment
        marginal_loglik[1] = 0.0;
    
        // Dynamic programming over segment ends
        for (t2 in 2:(n_time_points + 1)) {
            vector[t2 - 1] path_lls;
    
            // Iterate over all previous segmentations ending at t1
            for (t1 in 1:(t2 - 1)) {
                real prev_marginal = marginal_loglik[t1];
    
                real lp_cp_cur = t1 > 1 ? lp_cp[t1 - 1] : 0.0;
                real lp_no_cp_cur = t1 < (t2 - 2) ? sum(lp1m_cp[t1:(t2 - 2)]) : 0.0;
    
                real ll_segment = segments_loglik[t1, t2 - 1];
                real lp_segment_change = t1 > 1 ? lp_change_prior[t2 - 1] : 0.0;
    
                // Total log-prob for this segmentation path
                path_lls[t1] = prev_marginal + lp_cp_cur + lp_no_cp_cur + ll_segment + lp_segment_change;
            }
    
            // Aggregate over paths to compute marginal
            marginal_loglik[t2] = log_sum_exp(path_lls);
        }
    
        // Final log-marginal likelihood
        return marginal_loglik[n_time_points + 1];
  }

}


data {
  int<lower=1> n_time_points;
  int<lower=1> n_price_points;
  array[n_time_points, n_price_points] int histogram;
  vector[n_price_points] price_points;

}

transformed data {
  array[n_time_points, n_price_points] int histogram_cumulative = create_histogram_cumulative(histogram);
  matrix[n_time_points, n_time_points] segments_loglik = compute_segments_loglik(histogram_cumulative);
  vector[n_time_points] change_magnitudes = compute_change_magnitudes(histogram_cumulative, price_points);
}

parameters {
  vector<lower=0, upper=1>[n_time_points-1] cp_probs;

  // Prior hyperparameters
  real<lower=1> prior_change_mu;
  real<lower=0.0001> prior_change_sigma;
}

transformed parameters {
  vector[n_time_points] lp_change_prior = compute_lp_change_prior(change_magnitudes, prior_change_mu, prior_change_sigma);
}

model {
  // Optionally add prior over cp_probs_raw if desired, e.g.:
  cp_probs ~ beta(1, 5);
  prior_change_mu ~ normal(0, .5);
  prior_change_sigma ~ exponential(1);

  target += compute_marginal_loglik(cp_probs, segments_loglik, lp_change_prior);
}
