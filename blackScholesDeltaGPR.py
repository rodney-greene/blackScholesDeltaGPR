#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:29:43 2023

@author: rodneygreene
"""

#Use quantLib for the option valuation
import QuantLib as ql 

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

#Packages for Gaussian Process regression
import torch
import gpytorch


#Class implementation of the Gaussian Process prior. This model is used for all kernels except the spectral
#mixture kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    #Returns the GP prior. Allows for non-zero mean but usually the zero mea assumption is satisfactory. Recall
    # the von Mises conditioning on this prior (which is done by the gpytorch.likelihood object)
    #to obtain the predictive posterior obtains a non-zero mean
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#Create the training data. It's just theoretical european option values calculated with quantLib.
#One-year expiry european call struck at 100. 
today = ql.Date(5, ql.October, 2023)
ql.Settings.instance().evaluationDate = today
theStrike = 100.0
europeanCall = ql.EuropeanOption( ql.PlainVanillaPayoff(ql.Option.Call, theStrike), \
                              ql.EuropeanExercise( today+30 ) ) #Call with one-year expiry
    
#Set up market data handles
s = ql.SimpleQuote(100.0)
r = ql.SimpleQuote(0.05)
sigma = ql.SimpleQuote(0.20) #volatility
riskFreeCurve = ql.FlatForward(0, ql.TARGET(), ql.QuoteHandle(r), ql.Actual360())
volatility = ql.BlackConstantVol(0, ql.TARGET(), ql.QuoteHandle(sigma), ql.Actual360() )
process = ql.BlackScholesProcess(ql.QuoteHandle(s), 
                              ql.YieldTermStructureHandle(riskFreeCurve),
                              ql.BlackVolTermStructureHandle(volatility))

#Construct the pricing engine
engine = ql.AnalyticEuropeanEngine(process)
europeanCall.setPricingEngine(engine)

noiseVariance = 0.5  #This noise is a proxy for (bid/ask spread)**2 for the call option price
train_underlyingTensor = torch.linspace(80.0, 120.0, steps=200)  #The training grid of underlying stock price
test_underlyingTensor  = torch.linspace(80.0, 120.0, steps=40) #The test grid for the Delta calculations
def bsValue(underlying):
    s.setValue(underlying)
    return( europeanCall.NPV() + np.random.randn()*np.sqrt(noiseVariance) ) 
def bsDelta(underlying):
    s.setValue(underlying)
    return( europeanCall.delta() )

#In order to use apply_ we have to initialize train_valueTensor with the underlyings.
#Rember, the assignment operator is just a reference copy, so here in order to avoid polluting
#train_underlyingTensor we have to perform a deep copy. Likewise to use apply_ to get a tensor
#of theoretical Black-Scholes Delta's.
train_valueTensor = torch.clone(train_underlyingTensor)
theoreticalCallDeltaForTrainingDomain = torch.clone(train_underlyingTensor)
train_valueTensor.apply_(bsValue)
theoreticalCallDeltaForTrainingDomain.apply_(bsDelta)

# initialize likelihood and model with the RBF kernel
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_underlyingTensor, train_valueTensor, likelihood, gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))

training_iterations = 100000
model.train()
likelihood.train()
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_underlyingTensor)
    # Calc loss and backprop gradients
    loss = -mll(output, train_valueTensor)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

#Let's get the option Delta by centered finite difference. First let's get in the gradient disabled pyTorch context
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #While we are at it, let's get the regression results for call option value
    #over the training and test grids
    observed_predictionTrain = likelihood(model(train_underlyingTensor))
    observed_predictionTest = likelihood(model(test_underlyingTensor))
    
    test_underlyingTensorNoGrad = torch.clone(test_underlyingTensor)
    test_underlyingTensorNoGrad.requires_grad = False
    deltaUnderlying = 0.01
    test_underlyingTensorNoGrad_up = test_underlyingTensorNoGrad + deltaUnderlying
    test_underlyingTensorNoGrad_down = test_underlyingTensorNoGrad - deltaUnderlying
    observed_predictionUp = likelihood(model(test_underlyingTensorNoGrad_up))
    observed_predictionDown = likelihood(model(test_underlyingTensorNoGrad_down))
    deltaCenteredFiniteDifference = (observed_predictionUp.mean - observed_predictionDown.mean) / (2.0 * deltaUnderlying)
    
#Let's get option Delta using automatic differentiation. Note that here we are NOT in the gradient disabled pyTorch context - in 
#other words I do not us torch.no_grad()
#First step is to enable the using the Directed Acyclical Graph for computing the derivative by setting requires_grad = True
test_underlying_withAutogradTensor = torch.clone(test_underlyingTensor)
test_underlying_withAutogradTensor.requires_grad = True
observed_withAutogradPrediction = likelihood(model(test_underlying_withAutogradTensor)) #The y_train term in the GPR-Delta equation
theFunctionToDifferentiate = observed_withAutogradPrediction.mean.sum()
theFunctionToDifferentiate.backward()
optionDeltaWithAutoDifferentiationTensor = test_underlying_withAutogradTensor.grad  #auto-diff


#And finally let's calculate the option Delta using the correct GPR partial derivative using Equation (10) in 
# https://arxiv.org/abs/1901.11081
#First let's get the term [K + sigma^2 I]**(-1) y_train
trainCovarianceTensor = model.covar_module(train_underlyingTensor, train_underlyingTensor)
trainCovarianceTensor = trainCovarianceTensor.to_dense()
IdentityTensor = torch.eye(trainCovarianceTensor.size(dim=0), trainCovarianceTensor.size(dim=1) )
trainCovarianceTensor_positiveDefinite = trainCovarianceTensor + IdentityTensor*model.likelihood.noise.item()
#Get the lower-triangular Cholesky matrix of the positive definite covariance matrix
LTensor = torch.linalg.cholesky(trainCovarianceTensor_positiveDefinite)
#In order to use torch.cholesky_solve, train_valueTensor has to be a two dimensional tensor, so we reshape it
train_valueTensor_reshaped = train_valueTensor.reshape( [-1, 1])
#And finally we get [K + sigma^2 I]**(-1) y_train
covInverse_tngY = torch.cholesky_solve(train_valueTensor_reshaped, LTensor)

#Now let's get the derivative of the kernel evelauated on the test underlying grid
lengthScale = model.covar_module.base_kernel.lengthscale.item()
lengthScaleSqr = lengthScale * lengthScale
testTrainCovaianceTensor =  model.covar_module(test_underlyingTensor, train_underlyingTensor)
deltaKTest_KTrainTensor = test_underlyingTensor.reshape([-1, 1]) - train_underlyingTensor.reshape([1,-1])
dkernel_dTestTensor = (-1.0 / lengthScaleSqr) * deltaKTest_KTrainTensor * testTrainCovaianceTensor

#Let's put the pieces together to get the analytic GPR-Delta
correctDeltaTensor = torch.matmul(dkernel_dTestTensor, covInverse_tngY)

#Finally let's plot the results!
with torch.no_grad():
    # Initialize plot
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Get upper and lower confidence bounds
    lower, upper = observed_predictionTest.confidence_region()
    # Plot training data as black stars
    ax.plot(train_underlyingTensor.numpy(), train_valueTensor.numpy(), 'k*', markersize=1) #Training points
    # Plot predictive means as blue line
    ax.plot(test_underlyingTensor.numpy(), observed_predictionTest.mean.numpy(), 'b') #Prediction on test grid
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_underlyingTensor.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-3, 3])
    ax.legend(['Training Data', 'Test Predictions', 'Confidence Bounds'], fontsize=9) 
    ax.set_title('Call Option Value')
    ax.set_xlabel('Underlying Stock Price')
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Call Option Delta')
    ax.plot(train_underlyingTensor.numpy(),theoreticalCallDeltaForTrainingDomain.detach().numpy(), 'k-')
    ax.plot(test_underlyingTensor.detach().numpy(), deltaCenteredFiniteDifference.detach().numpy(), 'c--')
    ax.plot(test_underlyingTensor.detach().numpy(), correctDeltaTensor.detach().numpy(), 'g-.')
    ax.plot(test_underlyingTensor.detach().numpy(), optionDeltaWithAutoDifferentiationTensor.detach().numpy(), 'm-.')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(['BS Delta', 'Finite Diff', 'GPR-Correct', 'Auto-Diff'], fontsize=9) 
    ax.set_xlabel('Underlying Stock Price')    
    plt.show()


