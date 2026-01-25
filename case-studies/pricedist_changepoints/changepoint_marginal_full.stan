functions {
  
  array[] int array_row(array[,] int a, int index) {
      int n_row = dims(a)[1];
      int n_col = dims(a)[2];
      array[n_col] int row_;
      
      for (i in 1:n_col) {
        row_[i] = a[index, i];
      }
      
      return row_;
  }
  
  // Computes cumulative sum of the histogram over time
  array[,] int create_histogram_cumulative(array[,] int histogram)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      array[n_time_points, n_price_points] int result;
      
      // Initialize with first row
      for (p in 1:n_price_points)
        result[1, p] = histogram[1, p];

      // Reject if all values are zero in the entire histogram
      if( sum(histogram[1]) == 0 ) {
          reject("Row 1 in quantity histogram is all zeroes. Please exclude such rows.");
      }
      
      // Compute cumulative sum row-wise
      for (t in 2:n_time_points) {
          for (p in 1:n_price_points)
              result[t, p] = result[t-1, p] + histogram[t, p];
          
          // Check for empty histogram rows (no prices at all)
          if ( sum(histogram[t]) == 0 ) {
            reject("Row ", t, " in quantity histogram is all zeroes. Please exclude such rows.");
          }

      }
      return result;
  }

  // Extracts the histogram for a segment [t_start, t_end]
  array[] int extract_segment_histogram(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      array[n_price_points] int start_vals;
      array[n_price_points] int end_vals;
      array[n_price_points] int result;
      
      // Retrieve cumulative before the start
      int t_prev = t_start - 1;
      if (t_prev < 1) {
          for (i in 1:n_price_points) 
              start_vals[i] = 0;
      } else {
          start_vals = array_row(histogram_cumulative, t_prev);
      }

      // Compute end cumulative value
      end_vals = (t_end <= n_time_points) ? histogram_cumulative[t_end] :  histogram_cumulative[n_time_points];
      
      // Segment = difference between cumulative ends
      for (i in 1:n_price_points) 
          result[i] = end_vals[i] - start_vals[i];

      return result;
  }

  // Computes segment log-likelihood under multinomial model
  real compute_segment_loglik(int t_start, int t_end, array[,] int histogram_cumulative, vector cur_price_dist)
  {
      int n_price_points = dims(histogram_cumulative)[2];
      array[n_price_points] int histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative);
      real total_qty = sum(histogram);

      // Return multinomial log-likelihood using the provided price distribution
      real log_prob = multinomial_lpmf(histogram | cur_price_dist);
      return total_qty > 0 ? log_prob : 0.0;
  }

  // Computes segment log-likelihood under multinomial model
  real compute_segment_loglik_(int t_start, int t_end, array[,] int histogram_cumulative, vector cur_price_dist)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      array[n_price_points] int histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative);
      real total_qty = sum(histogram);
      vector[n_price_points] log_probs;
  
      // Compute log-probabilities for multinomial
      for (i in 1:n_price_points) {
          log_probs[i] = histogram[i] > 0 ? log(histogram[i] / total_qty) : 0.0;
      }

      // Return segment log-likelihood
      return total_qty > 0 ? dot_product( to_vector(histogram), log_probs) : 0.0;
  }

  // Precomputes all segment log-likelihoods for dynamic programming algo
  matrix compute_segments_loglik(array[,] int histogram_cumulative, array[,] vector price_dist) // 
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      matrix[n_time_points, n_time_points] segments_loglik;
  
      // Compute upper triangle log-likelihoods
      for (t1 in 1:n_time_points) {
          for (t2 in t1:n_time_points) {
              vector[n_price_points] cur_price_dist = price_dist[t1, t2];
              segments_loglik[t1, t2] = compute_segment_loglik(t1, t2, histogram_cumulative, cur_price_dist);
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
  
/*
  // Computes segment-by-segment difference statistic (e.g., mean price here) as a proxy for change magnitude
  // to-do: implement it by segment (with reasonable pruning)
  real compute_change_magnitude(array[] int histogram_prev, array[] int histogram_curr, vector price_points)
  {
      // Retrieve previous and current segment
      real mean_prev = dot_product( to_vector(histogram_prev), price_points) / sum(histogram_prev);
      real mean_curr = dot_product( to_vector(histogram_curr), price_points) / sum(histogram_curr);
      
      // Change magnitude is absolute mean difference
      return abs(mean_prev - mean_curr);
  }

  // Applies change magnitude prior to the change magnitude
  // These priors serve as a penalty or encouragement for placing changepoints
  vector compute_lp_change_prior(vector change_magnitude, real prior_mu, real prior_sigma)
  {
      int T = num_elements(change_magnitude);
      vector[T] lp;

      for (t in 1:T) {
          lp[t] = normal_lpdf( change_magnitude[t] | prior_mu, prior_sigma);
      }
  
      return lp;
  }
*/

  // Main forward pass algorithm: computes marginal log-likelihood via dynamic programming
  // Each time step t2 accumulates total log-probabilities over all segmentations ending at t2
  real compute_marginal_loglik(vector cp_probs_raw, matrix segments_loglik, array[,] 
                               vector price_dist, vector price_points, 
                               real prior_mu, real prior_sigma)
  {
        int n_time_points = cols(segments_loglik);
        int n_price_points = size(price_points);
        int n_cp = n_time_points - 1;
        real epsilon = 1e-12;

        // Clip cp_probs to avoid numerical errors
        vector[n_cp] cp_probs = fmin(fmax(cp_probs_raw, epsilon), 1 - epsilon);
        vector[n_cp] lp_cp = log(cp_probs);
        vector[n_cp] lp1m_cp = log1m(cp_probs);
        vector[n_time_points + 1] marginal_loglik;
    
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

                // move to separate function
                real lp_segment_change;
                if (t1 > 1) {
                    real mean_prev_segment = dot_product(price_dist[t1-1, t1-1], price_points);
                    real mean_curr_segment = dot_product(price_dist[t1, t2-1], price_points); 
                    real delta = abs(mean_curr_segment - mean_prev_segment);
                    lp_segment_change = normal_lpdf( delta | prior_mu, prior_sigma);

                } else {
                    lp_segment_change = 0.0;
                }
                // -end- move to separate function                
    
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
}

parameters {
  vector<lower=0, upper=1>[n_time_points-1] cp_probs;

  // Prior hyperparameters
  real<lower=1> prior_change_mu;
  real<lower=0.0001> prior_change_sigma;
  
  //matrix[n_time_points, n_time_points, n_price_points] histogram;
  array[ (n_time_points*(n_time_points+1) %/% 2) ] simplex[n_price_points] price_dist_list;
}

transformed parameters {
  //vector[n_time_points] lp_change_prior = compute_lp_change_prior(change_magnitudes, prior_change_mu, prior_change_sigma);
  array[n_time_points, n_time_points] simplex[n_price_points] price_dist;

  {
    vector[n_price_points] empty_simplex = rep_vector(1.0 / n_price_points, n_price_points);
    int idx_list = 1;
    for (t1 in 1:n_time_points) {
        for (t2 in 1:n_time_points) {
            if (t1 <= t2) {
                price_dist[t1, t2] = price_dist_list[idx_list];
                idx_list += 1;
            } else {
                price_dist[t1, t2] = empty_simplex;
            }
        }
    }
  }
  matrix[n_time_points, n_time_points] segments_loglik = compute_segments_loglik(histogram_cumulative, price_dist);
  //vector[n_time_points] change_magnitudes = compute_change_magnitudes(histogram_cumulative, price_points, prior_mu, prior_sigma);
}

model {
  // Optionally add prior over cp_probs_raw if desired, e.g.:
  cp_probs ~ beta(1, 5);
  prior_change_mu ~ normal(0, .5);
  prior_change_sigma ~ exponential(1);
  
  for (s in 1:(n_time_points * (n_time_points + 1) %/% 2)) {
    price_dist_list[s] ~ dirichlet(rep_vector(0.5, n_price_points));
  }
  // price_dist_vec ~ normal(0, 1);

  target += compute_marginal_loglik(cp_probs, segments_loglik, price_dist, price_points, prior_change_mu, prior_change_sigma);
}
