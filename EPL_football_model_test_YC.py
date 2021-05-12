#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:22:28 2020

@author: ashlee
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool
from datetime import datetime

def calc_means(param_dict, homeTeam, awayTeam):
    return [np.exp(param_dict['attack_'+homeTeam] + param_dict['defence_'+awayTeam] + param_dict['home_adv']),
            np.exp(param_dict['defence_'+homeTeam] + param_dict['attack_'+awayTeam])]

def rho_correction(x, y, lambda_x, mu_y, rho):
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

def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_goals=10):
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], params_dict['rho']) for away_goals in range(2)]
                                   for home_goals in range(2)])
    output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
    return output_matrix

def solve_parameters_decay(dataset, xi=0.001, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("something not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                      np.random.uniform(0,-1,(n_teams)), # defence strength
                                      np.array([0,1.0]) # rho (score correction), gamma (home advantage)
                                     ))
        
    def dc_log_like_decay(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma, t, xi=xi):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return  np.exp(-xi*t) * (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) + 
                                  np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

    def estimate_paramters(params):
        score_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
        rho, gamma = params[-2:]
        log_like = [dc_log_like_decay(row.HomeYellowC, row.AwayYellowC, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                                      score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], 
                                      rho, gamma, row.time_diff, xi=xi) for row in dataset.itertuples()]
        return -sum(log_like)
    opt_output = minimize(estimate_paramters, init_vals, options=options, constraints = constraints, **kwargs)
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["attack_"+team for team in teams] + 
                        ["defence_"+team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x))
  
    
epl_all = pd.DataFrame()
for year in range(17,21):
    epl_all = pd.concat((epl_all, pd.read_csv("http://www.football-data.co.uk/mmz4281/{}{}/E0.csv".format(year, year+1, sort=True))))
epl_all['Date'] = pd.to_datetime(epl_all['Date'],  format='%d/%m/%Y')
epl_all['time_diff'] = (max(epl_all['Date']) - epl_all['Date']).dt.days
epl_YC_1720 = epl_all[['HomeTeam','AwayTeam','HY','AY', 'FTR', 'time_diff']]
epl_YC_1720 = epl_YC_1720.rename(columns={'HY': 'HomeYellowC', 'AY': 'AwayYellowC'})
epl_YC_1720 = epl_YC_1720.dropna(how='all')

"""dates_df = pd.read_csv('Dates.csv')

list_dates = dates_df['2019-01-01'].tolist()

list_dates_test = ['2021-03-23']#, '2021-03-30', '2021-04-06']

for date in list_dates_test:
    # create dataset based on the dates list
    epl_1720_test = epl_1720[epl_1720['Date'] <= date]
    
    # calculate the params based on the above dataset
    test_params = solve_parameters_decay(epl_1720_test, xi = 0.00325)
    
    #create odds based on list of home teams and away teams for the next 10 games
    epl_test_prediction = pd.DataFrame(columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', 'BTTS', 'No BTTS'])
    
    # List of home teams and away teams that play eacha other.
    # Need to see whether I have to do this manually or whether I can code it 
    
    HomeTeam = ['Chelsea']
    AwayTeam = ['Brighton']

    # For the teams in the two lists above calculate the odds
    for i, j in zip(HomeTeam, AwayTeam):
        matrix = dixon_coles_simulate_match(test_params, i, j, max_goals=10)
    
        matrix_df = pd.DataFrame(matrix)
        matrix_df = matrix_df * 100
        
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        home_odds = round(1/home_win, 2)
        draw_odds = round(1/draw, 2)
        away_odds = round(1/away_win, 2)
        
        matrix_df.loc['Total', :] = matrix_df.sum(axis = 0)
        matrix_df.loc[:, 'Total'] = matrix_df.sum(axis = 1)
        
        not_btts = (matrix_df.iloc[-1, 0] + matrix_df.iloc[0, -1] - matrix_df.iloc[0, 0])
        btts = 100 - not_btts
        
        not_btts_odds = round(100/not_btts, 2)
        btts_odds = round(100/btts, 2)   
        
        home_away = [i, j, home_odds, draw_odds, away_odds, btts_odds, not_btts_odds]
        home_away_df = pd.DataFrame(home_away)
        home_away_trans = home_away_df.transpose()
        home_away_trans.columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                          'Away win', 'BTTS', 'No BTTS']
        epl_test_prediction = epl_test_prediction.append(home_away_trans)
        
        # Final dataset that contains the odds from Bet365 and odds calcualted here        
        final_dataset = epl_all[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 
                           'FTHG', 'FTAG', 'FTR', 'time_diff']]
        
        # Keeping it to the 2020 season
        epl_this_season_backseries = final_dataset[final_dataset['Date'] > '2020-09-12']

        # Merge the two datasets together to compare
        epl_this_season_backseries_test = pd.merge(epl_this_season_backseries, epl_test_prediction,
                                                   how = 'left', left_on=['HomeTeam', 'AwayTeam'],
                                                   right_on = ['HomeTeam', 'AwayTeam'])
        
   """    


"""final_dataset = epl_all[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 
                           'FTHG', 'FTAG', 'FTR', 'time_diff']]

final_dataset['Home_pred'] = ''
final_dataset['Draw_pred'] = ''
final_dataset['Away_pred'] = ''
final_dataset['odd_diff_home'] = ''
final_dataset['odd_diff_draw'] = ''
final_dataset['odd_diff_away'] = ''
final_dataset['result_chosen'] = ''
final_dataset['amount_bet'] = ''
final_dataset['outcome_of_bet'] = ''
final_dataset['profit_loss'] = ''
final_dataset['%_profit_loss'] = ''"""

      
params_YC = solve_parameters_decay(epl_YC_1720, xi = 0.00325)

epl_YC_prediction = pd.DataFrame(columns = ['HomeTeam', 'AwayTeam', 'O2.5YC', 'U2.5YC',
                                            'Home more cards', 'Draw', 'Away more cards'
                                            ])
HomeTeam = ['Leicester']
AwayTeam = ['Newcastle']
    
for i, j in zip(HomeTeam, AwayTeam):
    matrix_YC = dixon_coles_simulate_match(params_YC, i, j, max_goals=10)
    
    matrix_YC_df = pd.DataFrame(matrix_YC)
    matrix_YC_df = matrix_YC_df * 100
    
    home_win_YC = np.sum(np.tril(matrix_YC, -1))
    draw_YC = np.sum(np.diag(matrix_YC))
    away_win_YC = np.sum(np.triu(matrix_YC, 1))
    
    home_YC_odds = round(1/home_win_YC, 2)
    draw_YC_odds = round(1/draw_YC, 2)
    away_YC_odds = round(1/away_win_YC, 2)
    
    matrix_YC_df.loc['Total', :] = matrix_YC_df.sum(axis = 0)
    matrix_YC_df.loc[:, 'Total'] = matrix_YC_df.sum(axis = 1)
    
    U2_5YC = (matrix_YC_df.iloc[0, 0] + matrix_YC_df.iloc[0, 1]
             + matrix_YC_df.iloc[0, 2] + matrix_YC_df.iloc[1, 0]
             + matrix_YC_df.iloc[2, 0] + matrix_YC_df.iloc[1, 1])
    O2_5YC = 100 - U2_5YC
    
    U2_5YC_odds = round(100/U2_5YC, 2)
    O2_5YC_odds = round(100/O2_5YC, 2)   
    
    home_away = [i, j, O2_5YC_odds, U2_5YC_odds, 
                 home_YC_odds, draw_YC_odds, away_YC_odds]
    home_away_df = pd.DataFrame(home_away)
    home_away_trans = home_away_df.transpose()
    home_away_trans.columns = ['HomeTeam', 'AwayTeam', 'O2.5YC', 'U2.5YC', 
                               'Home more cards', 'Draw', 'Away more cards']
    
    epl_YC_prediction = epl_YC_prediction.append(home_away_trans)
    
#end_date = '2020-09-11'
#epl_backseries_for_2020 = final_dataset[final_dataset['Date'] <= end_date]  


#test edit for github
#test 2 for github
#testing for yellow card
