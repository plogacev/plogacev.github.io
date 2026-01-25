  be modular components within larger hierarchical models for retail analytics. 
  They can be used to detect structural changes in pricing regimes based on price histograms over time when the pricing of a product undergoes
  a meaningful change — such as list price changes, promotions, or changes in the availability of discounts. 

  The notebook in this folder present two implementations of a Bayesian changepoint detection model applied to synthetic sales data, where product prices change over time. The goal is to detect structural changes in pricing regime based on price histograms over time when the pricing of a product undergoes a meaningful change — such as list price changes, promotions, or changes in the availability of discounts.

  We assume that the distribution of prices for a product on any given day depends on the underlying price regime, which changes infrequently. While the price regime (list price and available discounts) remains stable for extended periods, the daily average price can fluctuate substantially due to varying proportions of discounted sales. Therefore, using the average price within a stable regime as a predictor is likely more informative than using volatile daily average prices.


The model determines pricing regimes based on daily price histograms (i.e., distributions of sold units across different price points). 
The first implementation, written in Rcpp, uses discrete parameters. The parameter estimation is carried out using a genetic algorithm implementation in R.
The second implementation marginalizes out the discrete segmentation parameters and estimates probabilities of specific changepoint locations.
Both are designed to serve as a modular component within larger hierarchical models for retail analytics.



  
  -------------------------------------------------------------------------------------------------------------------------------------
  
  
  ## Implementation 2
  
  - We implement the model as described above. For computational simplicity, we estimate the parameters $\tau_k$ as proportions of each segment. Specifically, for each segment $k$, the parameter $\tau_k$ is estimated as the relative proportions of the respective price points within that segment:
  
  $$
  \widehat{\tau}_k = \frac{ y_k }{ \sum_{p=1}^{P} y_{k[j]} },
$$
  , where $y_k$ stands for the price histogram over segment $k$, and $P$ is the price point index.


## Implementation 2

The model marginalizes over all possible segmentations of the time series, using a dynamic programming algorithm to recursively compute the total marginal likelihood across all changepoint configurations. It is well-suited for integration in hierarchical settings where multiple products or locations share changepoint structure. This is especially useful when you want to detect coordinated changes across multiple products or stores.


---

This portfolio project demonstrates a Bayesian changepoint detection algorithm for time series of price histograms, specifically tailored for transactional sales data with varying price regimes.

In many pricing applications, especially in retail and e-commerce, prices are not observed as single-valued time series but as distributions: a product might sell at different prices on the same day due to discounts, negotiated deals, or channel differences. Instead of modeling average price or revenue, this approach captures the full distribution of prices over time by discretizing them into histograms.