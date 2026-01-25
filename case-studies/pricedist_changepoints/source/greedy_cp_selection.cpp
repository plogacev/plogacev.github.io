#include <Rcpp.h>
#include <stdio.h>
using namespace Rcpp;

double log1m_exp(double x)
{
    if (x >= 0.0) stop("log1m_exp is undefined for x >= 0");
    if (x < -0.6931471805599453) {
        return std::log(1.0 - std::exp(x));
      
    } else {
        return std::log1p(-std::exp(x));
    }
}

// [[Rcpp::export]]
NumericMatrix create_histogram_cumulative(const IntegerMatrix& histogram)
{
  int n_time_points = histogram.nrow();
  int n_price_points = histogram.ncol();
  NumericMatrix histogram_cumulative(n_time_points, n_price_points);
  
  for (int p = 0; p < n_price_points; p++) {
      histogram_cumulative(0, p) = histogram(0, p);
  }
  
  for (int t = 1; t < n_time_points; t++)
  {
      int cur_row_sum = 0;
      for (int p = 0; p < n_price_points; p++) {
        histogram_cumulative(t, p) = histogram_cumulative(t - 1, p) + histogram(t, p);
        cur_row_sum += histogram(t, p);
      }
      
      if (cur_row_sum == 0) {
          stop("Row " + std::to_string(t) + " in histogram is all zero.");
      }
  }
  
  return histogram_cumulative;
}

// [[Rcpp::export]]
IntegerVector extract_segment_histogram(int t_start, int t_end, // assumes R-indexing
                                        const IntegerMatrix& histogram_cumulative)
{
    int n_time_points = histogram_cumulative.nrow();
    int n_price_points = histogram_cumulative.ncol();
    assert(t_start >= 1 && t_start <= n_time_points);
    assert(t_end >= 1 && t_end <= n_time_points);
    assert(t_start <= n_time_points);

    IntegerVector histogram_segment(n_price_points);
    
    int idx_start = (t_start - 1);
    int idx_end = (t_end - 1);
    if ( idx_start == 0 ) {
        histogram_segment = histogram_cumulative.row(idx_end);

    } else {
        IntegerMatrix::ConstRow start_vals = histogram_cumulative.row(idx_start-1);
        IntegerMatrix::ConstRow end_vals = histogram_cumulative.row(idx_end);
        histogram_segment = end_vals - start_vals;
    }

    return histogram_segment;
}

// [[Rcpp::export]]
double compute_segment_loglik(int t_start, int t_end, // assumes R-indexing
                              const IntegerMatrix& histogram_cumulative,
                              double epsilon = 1e-10) 
{
    int n_time_points = histogram_cumulative.nrow();
    int n_price_points = histogram_cumulative.ncol();
    assert(t_start >= 1 && t_start <= n_time_points);
    assert(t_end >= 1 && t_end <= n_time_points);
    assert(t_start <= n_time_points);

    NumericVector histogram_segment = (NumericVector)extract_segment_histogram(t_start, t_end, histogram_cumulative);

    double total = 0.0;
    for (int p = 0; p < n_price_points; p++) {
        histogram_segment[p] += epsilon;
        total += histogram_segment[p];
    }
    
    double loglik = 0.0;
    for (int p = 0; p < n_price_points; p++) {
        loglik += histogram_segment[p] * std::log(histogram_segment[p] / total);
    }

    return loglik;
}

// [[Rcpp::export]]
double compute_segmentation_loglik(const IntegerMatrix& histogram_cumulative,
                                   const LogicalVector& changepoints,
                                   const double log_lambda)
{
    int n_time_points = histogram_cumulative.nrow();
    if (changepoints.size() != n_time_points - 1) {
        stop("changepoints must have length T - 1");
    }

    double segments_loglik = 0.0;
    int t_start = 1; // uses R-indexing for function calls, and C-indexing internally, because the other functions assumes R-indexing as well
    for (int t = 1; t < n_time_points; t++)
    {
        if (changepoints[t-1]) {
            segments_loglik += compute_segment_loglik(t_start, t, histogram_cumulative);
            t_start = t + 1;
        }
    }
    // include the final segment
    segments_loglik += compute_segment_loglik(t_start, n_time_points, histogram_cumulative);
    
    // apply the penalty for the number of changepoints
    double prior = 0.0;
    if (!std::isnan(log_lambda)) {
        double log_1m_lambda = log1m_exp(log_lambda);
        prior = log_1m_lambda * std::count(changepoints.begin(), changepoints.end(), false) +
                   log_lambda * std::count(changepoints.begin(), changepoints.end(), true);
    }

    return (segments_loglik + prior);
}

// [[Rcpp::export]]
List locate_next_best_changepoint(const IntegerMatrix& histogram_cumulative,
                                  const LogicalVector& changepoints,
                                  const double log_lambda)
{
  int n_time_points = histogram_cumulative.nrow();
  if (changepoints.size() != n_time_points - 1)
      stop("changepoints must have length T - 1");
  
  double baseline_loglik = compute_segmentation_loglik(histogram_cumulative, changepoints, log_lambda);
  double max_loglik = R_NegInf;
  int max_loglik_index = -1;
  // LogicalVector best_changepoints = clone(changepoints);
  LogicalVector changepoints_trial = clone(changepoints);
  
  for (int t = 0; t < n_time_points - 1; t++) {
      if (!changepoints[t]) {
          changepoints_trial[t] = true;
          double cur_loglik = compute_segmentation_loglik(histogram_cumulative, changepoints_trial, log_lambda);
          if (cur_loglik > max_loglik) {
              max_loglik = cur_loglik;
              max_loglik_index = t;
          }
          changepoints_trial[t] = false;
      }
  }
  
  changepoints_trial[max_loglik_index] = true;
  
  return List::create(
    Named("changepoints") = changepoints_trial,
    Named("position") = (max_loglik_index + 1),
    Named("loglik") = max_loglik,
    Named("loglik_increment") = (max_loglik - baseline_loglik)
  );
}


List _locate_changepoints(const IntegerMatrix& histogram_cumulative,
                          const LogicalVector& changepoints,
                          const double log_lambda)
{
  int n_time_points = histogram_cumulative.nrow();
  
  double max_loglik = R_NegInf;
  LogicalVector changepoints_new(changepoints);
  List result;
  
  while (true) {
      result = locate_next_best_changepoint(histogram_cumulative, changepoints_new, log_lambda);
      double cur_loglik_increment = result["loglik_increment"];
      if (cur_loglik_increment < 0) {
          break;
      }
      
      changepoints_new = result["changepoints"];
      max_loglik = result["loglik"];
  }

  return List::create(
      Named("changepoints") = changepoints_new,
      Named("loglik") = max_loglik
  );
}


// [[Rcpp::export]]
List locate_changepoints(const IntegerMatrix& histogram_cumulative,
                         const double log_lambda)
{
    int n_time_points = histogram_cumulative.nrow();
    LogicalVector changepoints(n_time_points - 1, false);

    return _locate_changepoints(histogram_cumulative, changepoints, log_lambda);
}

List _reeval_changepoints(const IntegerMatrix& histogram_cumulative,
                         const LogicalVector& changepoints,
                         const double log_lambda)
{
    int n_time_points = histogram_cumulative.nrow();
    LogicalVector changepoints_new = clone(changepoints);
    double loglik = R_NegInf;
  
    for (int t = 0; t < n_time_points - 1; t++) {
        if (changepoints_new[t]) {
            changepoints_new[t] = false;
      
            List result = _locate_changepoints(histogram_cumulative, changepoints_new, log_lambda);
            changepoints_new = result["changepoints"];
            loglik = result["loglik"];
        }
    }
    
    return List::create(
        Named("changepoints") = changepoints_new,
        Named("loglik") = compute_segmentation_loglik(histogram_cumulative, changepoints_new, log_lambda)
    );
}

// [[Rcpp::export]]
List reeval_changepoints(const IntegerMatrix& histogram_cumulative,
                         const LogicalVector& changepoints,
                         const double log_lambda)
{
  LogicalVector changepoints_new(changepoints);
  
  List result = _reeval_changepoints(histogram_cumulative, changepoints, log_lambda);
  changepoints_new = result["changepoints"];
  double loglik = result["loglik"];
  
  return List::create(
      Named("changepoints") = changepoints_new,
      Named("loglik") = compute_segmentation_loglik(histogram_cumulative, changepoints_new, log_lambda)
  );

}

