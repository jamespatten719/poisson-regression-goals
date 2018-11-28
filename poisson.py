# -*- coding: utf-8 -*-
"""
Poisson Regression Goals

@author: jamespatten1996@gmail.com
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

# =============================================================================
# Parse JSON from football-data.org API
# =============================================================================

class scores(object):
    def __init__(self):
        """
        class to convert football-data.org dicts into pd dataframe format
        """
        self.headers = {'X-Auth-Token': '8ec939d825ec4600a775d4e1e0c8f6da'}
        self.r = requests.get('https://api.football-data.org/v2/competitions/PL/matches',headers=self.headers)
        self.plJSON = self.r.json()
        self.matchList =[]
    
    def getScores(self):
        """
        create a pandas dataframe row containing the team names,
        goals scored for each time and the matchday number
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
            
            #merged dict
            match = {**homeTeam, **awayTeam, **fullTime, **matchday}
            self.matchList.append(pd.DataFrame(match,index=[i]))
        return self.matchList

# =============================================================================
# poission regression model
# =============================================================================
df= pd.DataFrame(columns=['awayGoals', 'awayTeam', 'homeGoals', 'homeTeam',  'matchday'])
df = df.append(scores().getScores())
df = df[['homeTeam','homeGoals','awayTeam','awayGoals']]

# split Data Frame into past and future GW
df_past = df.loc[df['homeGoals'] >= 0]
df_future= df.loc[df['homeGoals'].isnull()]

# work out poisson probabilities of goal differences between home and away team of - 8 to plus 8
skellam_pred = [skellam.pmf(i,  df_past['homeGoals'].mean(),  df_past['awayGoals'].mean()) for i in range(-8,8)]

# poisson regression

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

poisson_model.predict(pd.DataFrame(data={'team': 'Arsenal FC', 'opponent': 'Tottenham Hotspur FC',
                                       'home':1},index=[1]))
    
poisson_model.predict(pd.DataFrame(data={'team': 'Tottenham Hotspur FC', 'opponent': 'Arsenal FC',
                                       'home':1},index=[1]))