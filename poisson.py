# -*- coding: utf-8 -*-
"""
Poisson Regression Goals

@author: jamespatten1996@gmail.com
implementation from: dashee87.github.io
"""

# =============================================================================
# import libs
# =============================================================================
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson,skellam
# poisson regression 
import statsmodels.api as sm
import statsmodels.formula.api as smf
# calculus
from scipy.optimize import minimize


# =============================================================================
# Parse JSON from football-data.org API
# =============================================================================

class football_data(object):
    """
    Class to make requests from the football-data.org
    API
    """
    def __init__(self):
        self.headers = {'X-Auth-Token': '8ec939d825ec4600a775d4e1e0c8f6da'}
        self.r = requests.get('https://api.football-data.org/v2/competitions/PL/matches',headers=self.headers)
        self.plJSON = self.r.json()
        self.matchList =[]
    
    def getScores(self):
        """
        create a pandas dataframe row containing the team names,
        goals scored for each team, matchday number and matchday date
        """

        matches = self.plJSON['matches']
        for i in range(0,len(matches)):
            indMatch = matches[i]
            
            #score
            score = indMatch['score']
            fullTime = score['fullTime']
            fullTime['homeGoals'] = fullTime.pop('homeTeam')
            fullTime['awayGoals'] = fullTime.pop('awayTeam')
        
            #team names
            homeTeam = {'homeTeam': indMatch['homeTeam']['name']}
            awayTeam = {'awayTeam': indMatch['awayTeam']['name']}
        
            #matchday
            matchday = {'matchday':indMatch['matchday']}
            
            #matchday date
            matchDate = {'matchDate':indMatch['utcDate']}
            
            #merged dict
            match = {**homeTeam, **awayTeam, **fullTime, **matchday,**matchDate}
            self.matchList.append(pd.DataFrame(match,index=[i]))
        return self.matchList

# =============================================================================
# dataframe preparation
# =============================================================================
df= pd.DataFrame(columns=['awayGoals', 'awayTeam', 'homeGoals', 'homeTeam',  'matchday','matchDate'])
df = df.append(football_data().getScores())

# split Data Frame into past and future GW
df_past = df.loc[df['homeGoals'] >= 0]
df_future= df.loc[df['homeGoals'].isnull()]


df_past['matchDate'] = pd.to_datetime(df_past['matchDate'])
df_past['time_diff'] = (max(df_past['matchDate']) - df_past['matchDate']).dt.days
df_past = df_past[['homeTeam','homeGoals','awayTeam','awayGoals','time_diff']] 
df_past.head()
# =============================================================================
# poisson regression model 
# =============================================================================

# work out poisson probabilities of goal differences between home and away team of - 8 to plus 8
skellam_pred = [skellam.pmf(i,  df_past['homeGoals'].mean(),  df_past['awayGoals'].mean()) for i in range(-8,8)]

# restructure dataframe by splitting home and away fixtures
goal_model_data = pd.concat([df_past[['homeTeam','awayTeam','homeGoals']].assign(home=1).rename(
            columns={'homeTeam':'team', 'awayTeam':'opponent','homeGoals':'goals'}),
           df_past[['awayTeam','homeTeam','awayGoals']].assign(home=0).rename(
            columns={'awayTeam':'team', 'homeTeam':'opponent','awayGoals':'goals'})])

goal_model_data['goals'] = goal_model_data['goals'].astype(int)

#create generalised linear model 
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
#error as data is not in integer format 
poisson_model.summary()

poisson_model.predict(pd.DataFrame(data={'team': 'Manchester United FC', 'opponent': 'Arsenal',
                                       'home':1},index=[1]))

# =============================================================================
# Dixon-Coles implementation (creds: dashee87.github.io)
# =============================================================================
    
class dixons_coles(object):
    """
    Set a functions to implement the Dixons-Model
    Implementation from the excellent dashee87.github.io
    """
    def __init__(self,dataset,xi):
        """
        Instantiate variables
        """
        self.dataset = dataset 
        self.teams = np.sort(self.dataset['homeTeam'].unique())
        self.away_teams = np.sort(self.dataset['awayTeam'].unique())
        self.n_teams = len(self.teams)
        self.xi = xi
         
    def tau(self,x, y, lambda_x, mu_y, rho):
        """
        Tau function with rho dependency parameter to reduce
        understatement of low scoring matches
        """
        if x==0 and y==0:
            return 1- (lambda_x * mu_y * rho)
        elif x==0 and y==1:
            return 1 + (lambda_x * rho)
        elif x==1 and y==0:
            return 1 + (mu_y * rho)
        elif x==1 and y==1:
            return 1 - rho
        else:
            return 1.0
                         
    def estimate_paramters(self,params):
        """
        Creating Log-likelihood function to be minimised wrt parameters 
        """
        score_coefs = dict(zip(self.teams, params[:self.n_teams])) #get first half of init_values
        defend_coefs = dict(zip(self.teams, params[self.n_teams:(2*self.n_teams)])) #get 2nd half of init_values
        rho, gamma = params[-2:] #get last two values in init_values
        log_like = [self.dc_log_like_decay(row.homeGoals, row.awayGoals, score_coefs[row.homeTeam], defend_coefs[row.homeTeam],
                     score_coefs[row.awayTeam], defend_coefs[row.awayTeam], rho, gamma,row.time_diff, self.xi) for row in self.dataset.itertuples()]
        return -sum(log_like)
            
    def solve_parameters_decay(self, xi=0.001, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                         constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
        """
        Maximise log-liklihood function to solve for alpha, beta, 
        gamma and rho parameters w/constraint to prevent overparametirisation.
        Constraint says that all attacking coeff must sum to 1. If this is not given
        it will be imposible to find a unique set of params that minimise the
        log liklihood. 
        """
        if not np.array_equal(self.teams, self.away_teams):
            raise ValueError("something not right")
        if init_vals is None:
            # random initialisation of model parameters
            init_vals = np.concatenate((np.random.uniform(0,1,(self.n_teams)), # attack strength
                                          np.random.uniform(0,-1,(self.n_teams)), # defence strength
                                          np.array([0,1.0]) # rho (score correction), gamma (home advantage)
                                         ))
        opt_output = minimize(self.estimate_paramters, init_vals, options=options, constraints = constraints, **kwargs)
        if debug:
            # sort of hacky way to investigate the output of the optimisation process
            return opt_output
        else:
            return dict(zip(["attack_"+team for team in self.teams] + 
                            ["defence_"+team for team in self.teams] +
                            ['rho', 'home_adv'],
                            opt_output.x))
        
    def dc_log_like_decay(self,x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma, t, xi):
        """
        Setting up expected goals scored and conceded (lambda/my) to be 
        used in the log-liklihood function
        """
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return  np.exp(-xi*t) * (np.log(self.tau(x, y, lambda_x, mu_y, rho)) + 
                                  np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

# optimise parameters for a given value of xi
params = dixons_coles(df_past,0.0018).solve_parameters_decay()
