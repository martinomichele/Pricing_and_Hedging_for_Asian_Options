# Pricing and Hedging for Asian Options

### **Overview**
This project explores the application of quasi–Monte Carlo (QMC) methods to option pricing and hedging in the Black–Scholes framework.  
The study focuses on Asian call options, both geometric and arithmetic, and evaluates numerical efficiency, variance reduction, and dynamic hedging performance.  
The work is divided into four main components.

### **Project Components**
  - **Quasi–Monte Carlo Methods and GBM Constructions:**  
  Comparison of QMC and standard Monte Carlo approaches for pricing geometric Asian options.  
  Various geometric Brownian motion path constructions (standard, PCA, and Brownian Bridge) were tested for convergence speed and computational cost.

<p align="center">
<img src="/Outputs/output_3_1.png" width="400" height="400"><img src="/Outputs/output_3_2.png" width="400" height="400">
</p>

<p align="center">
<img src="/Outputs/output_3_3.png" width="400" height="400"><img src="/Outputs/output_3_4.png" width="400" height="400">
</p>

  - **Variance Reduction for Arithmetic Options:**  
  Implementation of antithetic variates and geometric control variates to improve pricing stability and reduce estimator noise when simulating arithmetic Asian options.

<p align="center">
<img src="/Outputs/output_11.png" width="400" height="400"><img src="/Outputs/output_10.png" width="400" height="400">
</p>

  - **Dynamic Delta–Hedging Simulations:**  
  Analysis of hedged and unhedged strategies for both option types.  
  The geometric case was handled analytically using closed-form price and conditional deltas, while the arithmetic case relied on RQMC estimates combined with variance reduction techniques for pricing and initial delta and use of geometric proxies for later deltas to improve computational cost.

<p align="center">
<img src="/Outputs/output_2_3.png" width="400" height="400"><img src="/Outputs/output_2_4.png" width="400" height="400">
</p>

<p align="center">
<img src="/Outputs/output_2_1.png" width="400" height="400"><img src="/Outputs/output_2_2.png" width="400" height="400">
</p>

  - **Real–World Data Application:**
  Extension of the methodology to historical market data using the $\texttt{yfinance}$ library.  
  Rolling volatilities were used for pricing and hedging, assuming a zero interest rate for simplicity.

<p align="center">
<img src="/Outputs/Output_4.png" width="600" height="600">
</p>

### **Key Findings**

  - QMC methods converge faster and more consistently than standard Monte Carlo simulations.  
  - The Brownian Bridge approach offers the best trade-off between speed and accuracy.  
  - Variance reduction through geometric control variate significantly improves pricing precision compared to antithetic variate.  
  - GBM simulations show that Delta–hedging mitigate large losses observed in unhedged strategies shrinking considerably the variance of the profits' distribution.  
  - Despite real market data deviating from theoretical assumptions, the proposed methods remain effective in practice.

### **References**
  - Caflisch, R. E. (1998). $\textit{Monte Carlo and quasi–Monte Carlo methods.}$
  - Joshi M.S. (2003), $\textit{The Concepts and Practice of Mathematical Finance.}$
  - Kemna A.G.Z., Vorst A.C,F. (1990), $\textit{A Pricing Method for Option Based on Average Asset Values.}$
  - Boyle P., Broadie M., Glasserman P. (1997), $\textit{Monte Carlo Methods for Security Pricing.}$
  - Kwok Y.K. (2008), $\textit{Mathematical Models of Financial Derivatives.}$
