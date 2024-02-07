import numpy as np
import torch
import random

from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import config_enumerate, SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta

import matplotlib.pyplot as plt

from scipy.stats import beta, expon, norm

import pickle
import pandas as pd


true_cJetParams = [4.8,7.4]
true_bJetParams = [2.9,1.2]

tagParams = np.stack([true_cJetParams, true_bJetParams])


def TruncatedExponential(lb, ub, rate=1.0):
    lb, ub, rate = broadcast_all(lb, ub, rate)
    return dist.TransformedDistribution(
        dist.Uniform((-rate * ub).exp(), (-rate * lb).exp()),
        [dist.transforms.ExpTransform().inv,
         dist.transforms.AffineTransform(loc=0, scale=-1/rate)]
    )

class TruncatedNormal(dist.Rejector):
    def __init__(self, loc, scale, min_x, max_x):
        propose = dist.Normal(loc, scale)

        def log_prob_accept(x):
            return (x > min_x).type_as(x).log()  +  (x < max_x).type_as(x).log()

        log_scale = torch.log(dist.Normal(loc, scale).cdf(max_x) - dist.Normal(loc, scale).cdf(min_x))
        super(TruncatedNormal, self).__init__(propose, log_prob_accept, log_scale)



mubb = 125
sigmabb = 7
lambdab = 0.004

lb = 75
ub = 175

signalDist = TruncatedNormal(loc = mubb, scale=sigmabb, min_x=torch.Tensor([lb]), max_x=torch.Tensor([ub]))
bkgDist = TruncatedExponential(lb=lb, ub=ub, rate=lambdab)


Nevents = 20000
S = 200

p4b = S/Nevents   # S/N
p2b2c = (Nevents-S)/(2*Nevents)


K = 3
J = 2

classMapScores = pyro.ops.indexing.Vindex(torch.tensor([[0,0,0,0],[1,1,1,1],[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0]]))
weightsForPermScores = pyro.ops.indexing.Vindex(torch.tensor([ [1]+7*[0] ,  [0,0]+6*[1./6] , [0]+[1]+6*[0]]))
priorParamsBetas = pyro.ops.indexing.Vindex(torch.tensor(tagParams).float())

classMapMass = pyro.ops.indexing.Vindex(torch.tensor([[False,False], [False, False], [True,True]]))

@config_enumerate
def model(data=torch.zeros(10,6)):
    weights = pyro.sample("latent_weights", dist.Dirichlet(torch.tensor(np.ones(K)).double()))

    with pyro.plate("jet_classes", J) as i:
        alpha = pyro.sample("latent_alpha", dist.Normal(loc=priorParamsBetas[i,0], scale=priorParamsBetas[i,0]*0.1))
        beta = pyro.sample("latent_beta", dist.Normal(loc=priorParamsBetas[i,1], scale=priorParamsBetas[i,1]*0.1))

    mu_gauss = pyro.sample("latent_mu", dist.Normal(mubb*1.05, scale=mubb*0.1))
    sigma_gauss = pyro.sample("latent_sigma", dist.Normal(sigmabb*0.90, scale=sigmabb*0.1))
    lambda_exp = pyro.sample("latent_lambda", dist.Uniform(low=0.01*lambdab,high=5*lambdab))

    with pyro.plate("data", data.size(0)):
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        permutationNrScores = pyro.sample("permutationScore", dist.Categorical(weightsForPermScores[assignment]))
        permutationScores = classMapScores[permutationNrScores]
       
        pyro.sample("scores", dist.Beta(alpha[permutationScores], beta[permutationScores]).to_event(1), obs=data[:,:4])        
        pyro.sample("mass", dist.MaskedMixture(
                            classMapMass[assignment], 
                            TruncatedExponential(lb=lb, ub=ub, rate=lambda_exp), 
                            TruncatedNormal(loc=mu_gauss, scale=sigma_gauss, min_x=torch.Tensor([lb]), max_x=torch.Tensor([ub])), 
                            validate_args=False).to_event(1), 
                                obs=data[:,4:])  

def init_loc_fn(site):
    if site["name"] == "latent_weights":
        # Initialize weights to uniform.
        site["constraint"]=constraints.simplex
        return torch.ones(K, dtype=torch.float) / K
    if site["name"] == "latent_alpha":
        site["constraint"]=constraints.positive
        return torch.tensor([true_cJetParams[0], true_bJetParams[0]], dtype=torch.float)*torch.rand(2)*10
    if site["name"] == "latent_beta":
        site["constraint"]=constraints.positive
        return torch.tensor([true_cJetParams[1], true_bJetParams[1]], dtype=torch.float)*torch.rand(2)*10
    if site["name"] == "latent_mu":
        return (torch.rand(1, dtype=torch.float)*0.6+0.7)*mubb
    if site["name"] == "latent_sigma":
        # site["constraint"]=constraints.positive
        return (torch.rand(1, dtype=torch.float)*0.6+0.7)*sigmabb
    if site["name"] == "latent_lambda":
        site["constraint"]=constraints.positive
        return (torch.rand(1, dtype=torch.float)*0.6+0.7)*lambdab
    raise ValueError(site["name"])


for n in range(25):
    labels = []  # 0: 4c,exp ; 1: 2b2c,exp ; 2: 4b,gauss
    data = []

    np.random.seed(n)
    torch.manual_seed(n)

    for i in range(Nevents):
        test = np.random.random()

        if test < p4b:
            labels.append(2)
            scoreSample = beta.rvs(*true_bJetParams, size=4)
            massSample = signalDist.sample([2]).numpy()
        elif test < p4b+p2b2c:
            labels.append(1)
            scoreSample = np.append(beta.rvs(*true_cJetParams, size=2), beta.rvs(*true_bJetParams, size=2))
            random.shuffle(scoreSample)
            massSample = bkgDist.sample([2]).numpy()
        else:
            labels.append(0)
            scoreSample = beta.rvs(*true_cJetParams, size=4)
            massSample = bkgDist.sample([2]).numpy()

        event = np.append(scoreSample, massSample)
        data.append(event)

    labels = np.array(labels)
    data = np.array(data)

    run_data = {
        "true_mubb": mubb,
        "true_sigmabb": sigmabb,
        "true_lambdab": lambdab,
        "true_bJetParams": true_bJetParams,
        "true_cJetParams": true_cJetParams,
        "true_pS": p4b,
        "true_p2b2c": p2b2c,
        "lb": lb,
        "ub": ub
    }

    data_train = torch.tensor(data.astype(np.float32))

    model(data_train)

    num_steps = 10000
    initial_lr = 0.1
    gamma = 0.01  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / num_steps)

    optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    # optim = pyro.optim.ClippedAdam({'lr': 0.001})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    def initialize(seed):
        global global_guide, svi
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        global_guide = AutoDelta(poutine.block(model, expose=['latent_weights', 'latent_alpha', 'latent_beta', 'latent_mu','latent_sigma','latent_lambda']), init_loc_fn=init_loc_fn)
        svi = SVI(model, global_guide, optim, loss=elbo)
        steps=500
        for nstep in range(steps):
            svi.step(data_train)
        return svi.evaluate_loss(data_train)
    
    print(f"##### RUN {n} #####")
    loss, seed = min((initialize(seed), seed) for seed in range(100))
    loss = initialize(seed)

    print('seed = {}, initial_loss = {}'.format(seed, loss))

    losses = []
    for i in range(num_steps):
        loss = svi.step(data_train)
        losses.append(loss)
        if not i%1000:
            print(loss)

    print(loss)

    map_estimates = global_guide(data_train)

    run_data.update(map_estimates)

    with open(f"run_data_{n}.pkl", "wb") as rf:
        pickle.dump(run_data, rf)

    np.save(f"data_{n}.npy", data)
    np.save(f"labels_{n}.npy", labels)