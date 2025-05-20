import pytest
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'src', 'options')
sys.path.append(module_path)

from black_scholes import BlackScholes

# Test data
@pytest.fixture
def sample_option():
    return BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

@pytest.fixture
def itm_call():
    return BlackScholes(S=110, K=100, T=1.0, r=0.05, sigma=0.2)

@pytest.fixture
def otm_call():
    return BlackScholes(S=90, K=100, T=1.0, r=0.05, sigma=0.2)

@pytest.fixture
def atm_option():
    return BlackScholes(S=100, K=100, T=0.5, r=0.05, sigma=0.2)

def test_d1_d2(sample_option):
    """Test d1 and d2 calculations"""
    d1 = sample_option.d1()
    d2 = sample_option.d2()
    
    # For ATM option with T=1, r=0.05, sigma=0.2
    expected_d1 = 0.35
    expected_d2 = 0.15
    
    assert abs(d1 - expected_d1) < 0.0001
    assert abs(d2 - expected_d2) < 0.0001
    assert d1 - d2 == pytest.approx(sample_option.sigma * np.sqrt(sample_option.T))

def test_european_call_price(sample_option, itm_call, otm_call):
    """Test European call option pricing"""
    # ATM option
    atm_price = sample_option.price_european_call()
    assert atm_price > 0
    
    # ITM option
    itm_price = itm_call.price_european_call()
    assert itm_price > atm_price
    
    # OTM option
    otm_price = otm_call.price_european_call()
    assert otm_price < atm_price
    
    # Test expiration (T=0)
    expiring_option = BlackScholes(S=100, K=100, T=0, r=0.05, sigma=0.2)
    assert expiring_option.price_european_call() == 0

def test_european_put_price(sample_option, itm_call, otm_call):
    """Test European put option pricing"""
    # ATM option
    atm_price = sample_option.price_european_put()
    assert atm_price > 0
    
    # ITM put (OTM call)
    itm_put_price = otm_call.price_european_put()
    assert itm_put_price > atm_price
    
    # OTM put (ITM call)
    otm_put_price = itm_call.price_european_put()
    assert otm_put_price < atm_price
    
    # Test expiration (T=0)
    expiring_option = BlackScholes(S=100, K=100, T=0, r=0.05, sigma=0.2)
    assert expiring_option.price_european_put() == 0

def test_delta(sample_option, itm_call, otm_call):
    """Test option delta calculations"""
    # Call deltas
    atm_call_delta = sample_option.delta_european_call()
    itm_call_delta = itm_call.delta_european_put()
    otm_call_delta = otm_call.delta_european_call()
    
    assert 0 < atm_call_delta < 1
    assert itm_call_delta < atm_call_delta
    assert otm_call_delta < atm_call_delta
    
    # Put deltas
    atm_put_delta = sample_option.delta_european_put()
    assert -1 < atm_put_delta < 0
    
    # Put-Call parity for deltas
    assert atm_call_delta - atm_put_delta == pytest.approx(1)

def test_gamma(sample_option, atm_option):
    """Test option gamma calculations"""
    atm_gamma = sample_option.gamma()
    shorter_term_gamma = atm_option.gamma()
    
    assert atm_gamma > 0
    assert shorter_term_gamma > atm_gamma  # Gamma increases as time to expiration decreases

def test_theta(sample_option):
    """Test option theta calculations"""
    call_theta = sample_option.theta_european_call()
    put_theta = sample_option.theta_european_put()
    
    assert call_theta < 0
    assert put_theta < 0

def test_vega(sample_option, atm_option):
    """Test option vega calculations"""
    atm_vega = sample_option.vega()
    shorter_term_vega = atm_option.vega()
    
    assert atm_vega > 0
    assert atm_vega > shorter_term_vega  # Vega decreases as time to expiration decreases

def test_rho(sample_option):
    """Test option rho calculations"""
    call_rho = sample_option.rho_european_call()
    put_rho = sample_option.rho_european_put()
    
    assert call_rho > 0  # Call rho is positive
    assert put_rho < 0  # Put rho is negative

def test_put_call_parity(sample_option):
    """Test put-call parity"""
    call_price = sample_option.price_european_call()
    put_price = sample_option.price_european_put()
    forward_price = sample_option.S - sample_option.K * np.exp(-sample_option.r * sample_option.T)
    
    assert abs(call_price - put_price - forward_price) < 1e-10 