import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        '''
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        '''
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self) -> float:
        """Calculate d1 parameter for Black-Scholes formula.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        return (np.log(self.S/self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self) -> float:
        """Calculate d2 parameter for Black-Scholes formula.

        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        return BlackScholes.d1(self) - self.sigma * np.sqrt(self.T)

    def price_european_call(self) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Returns:
        float: European call option price.
        """
        d1 = self.d1()
        d2 = self.d2()

        if self.T == 0:
            return max(0, self.S - self.K)
        else:
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def price_european_put(self) -> float:
        """
        Calculate European put option price using Black-Scholes formula.

        Returns:
        float: European put option price.
        """
        d1 = self.d1()
        d2 = self.d2()

        if self.T == 0:
            return max(0, self.K - self.S)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta_european_call(self) -> float:
        """Calculate call option delta
        
        Returns:
        float: Delta of the call option.
        """
        return norm.cdf(self.d1())

    def delta_european_put(self) -> float:
        """Calculate put option delta
        
        Returns:
        float: Delta of the put option.
        """
        return norm.cdf(self.d1()) - 1

    def gamma(self) -> float:
        """Calculate option gamma.
        
        Returns:
        float: Gamma of the option.
        """
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_european_call(self) -> float:
        """Calculate call option theta.
        
        Returns:
        float: Theta of the call option.
        """
        d1 = self.d1()
        d2 = self.d2()
        return (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def theta_european_put(self) -> float:
        """Calculate put option theta.
        
        Returns:
        float: Theta of the put option.
        """
        d1 = self.d1()
        d2 = self.d2()
        return (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def vega(self) -> float:
        """Calculate option vega.
        
        Returns:
        float: Vega of the option.
        """
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1())

    def rho_european_call(self) -> float:
        """Calculate call option rho.
        
        Returns:
        float: Rho of the call option.
        """
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def rho_european_put(self) -> float:
        """Calculate put option rho.
        
        Returns:
        float: Rho of the put option.
        """
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) 