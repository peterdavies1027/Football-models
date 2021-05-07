#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:16:05 2020

@author: ashlee
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool

def calc_means(bund2_param_dict, homeTeam, awayTeam):
    return [np.exp(bund2_param_dict['attack_'+homeTeam] + bund2_param_dict['defence_'+awayTeam] + bund2_param_dict['home_adv']),
            np.exp(bund2_param_dict['defence_'+homeTeam] + bund2_param_dict['attack_'+awayTeam])]

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

def solve_parameters(dataset, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("Something's not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                      np.random.uniform(0,-1,(n_teams)), # defence strength
                                      np.array([0, 1.0]) # rho (score correction), gamma (home advantage)
                                     ))

    def dc_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) + 
                np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

    def estimate_paramters(bund2_params):
        score_coefs = dict(zip(teams, bund2_params[:n_teams]))
        defend_coefs = dict(zip(teams, bund2_params[n_teams:(2*n_teams)]))
        rho, gamma = bund2_params[-2:]
        log_like = [dc_log_like(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                     score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], rho, gamma) for row in dataset.itertuples()]
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

def dixon_coles_simulate_match(bund2_params_dict, homeTeam, awayTeam, max_goals=10):
    team_avgs = calc_means(bund2_params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], bund2_params_dict['rho']) for away_goals in range(2)]
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

    def estimate_paramters(bund2_params):
        score_coefs = dict(zip(teams, bund2_params[:n_teams]))
        defend_coefs = dict(zip(teams, bund2_params[n_teams:(2*n_teams)]))
        rho, gamma = bund2_params[-2:]
        log_like = [dc_log_like_decay(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
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

def get_1x2_probs(match_score_matrix):
    return dict({"H":np.sum(np.tril(match_score_matrix, -1)), 
                 "A":np.sum(np.triu(match_score_matrix, 1)), "D":np.sum(np.diag(match_score_matrix))})

def build_temp_model(dataset, time_diff, xi=0.000, init_params=None):
    test_dataset = dataset[((dataset['time_diff']<=time_diff) & (dataset['time_diff']>=time_diff-2))]
    if len(test_dataset)==0:
        return 0
    train_dataset = dataset[dataset['time_diff']>time_diff]
    train_dataset['time_diff'] = train_dataset['time_diff'] - time_diff
    bund2_params = solve_parameters_decay(train_dataset, xi=xi, init_vals=init_params)
    predictive_score = sum([np.log(get_1x2_probs(dixon_coles_simulate_match(
                    bund2_params, row.HomeTeam, row.AwayTeam))[row.FTR]) for row in test_dataset.itertuples()])
    return predictive_score    
    
def check_xi(match_day):
    xi_score = build_temp_model(bund2_1820, match_day, xi=my_xi)
    return xi_score

def bund2_match_day(HomeTeam, AwayTeam):
    matrix = dixon_coles_simulate_match(bund2_params, HomeTeam, AwayTeam, max_goals=10)
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    home_odds = 1/home_win
    draw_odds = 1/draw
    away_odds = 1/away_win
    
    return matrix, home_win, draw, away_win, home_odds, draw_odds, away_odds
    
    
    
bund2_1820 = pd.DataFrame()
for year in range(18,20):
    bund2_1820 = pd.concat((bund2_1820, pd.read_csv("http://www.football-data.co.uk/mmz4281/{}{}/D2.csv".format(year, year+1, sort=True))))
bund2_1820['Date'] = pd.to_datetime(bund2_1820['Date'],  format='%d/%m/%Y')
bund2_1820['time_diff'] = (max(bund2_1820['Date']) - bund2_1820['Date']).dt.days
bund2_1820 = bund2_1820[['HomeTeam','AwayTeam','FTHG','FTAG', 'FTR', 'time_diff']]
bund2_1820 = bund2_1820.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
bund2_1820 = bund2_1820.dropna(how='all')

# value of xi you wish to test
my_xi = 0.00325
# skip days where no matches took place
#match_days = [day for day in range(99,-1,-3) if len(epl_1720[((epl_1720['time_diff']<=day) & (epl_1720['time_diff']>=(day-2)))])]
#if __name__ == '__main__':
#    print("Pooling...")
#    with Pool() as pool:         # start worker processes (number will depend on your computer's architecture)
#        xi_result = pool.map(check_xi, match_days)
#    with open('find_xi_5season_{}.txt'.format(str(my_xi)[2:]), 'wb') as thefile:
#        pickle.dump(xi_result, thefile)
        
bund2_params = solve_parameters_decay(bund2_1820, xi = 0.00325)

bund2_prediction = pd.DataFrame(columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 'No BTTS'])

HomeTeam = ['Hannover', 'Holstein Kiel']
AwayTeam = ['Darmstadt', 'St Pauli']

for i, j in zip(HomeTeam, AwayTeam):
    matrix = dixon_coles_simulate_match(bund2_params, i, j, max_goals=10)
    
    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df * 100
    
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    home_odds = round(1/home_win, 2)
    draw_odds = round(1/draw, 2)
    away_odds = round(1/away_win, 2)
    
    ho_dr = round(1/(home_win + draw), 2)
    dr_aw = round(1/(draw + away_win), 2)
    ha_win = round(1/(home_win + away_win), 2)
    
    matrix_df.loc['Total', :] = matrix_df.sum(axis = 0)
    matrix_df.loc[:, 'Total'] = matrix_df.sum(axis = 1)
    
    not_btts = (matrix_df.iloc[-1, 0] + matrix_df.iloc[0, -1] - matrix_df.iloc[0, 0])
    btts = 100 - not_btts
    
    not_btts_odds = round(100/not_btts, 2)
    btts_odds = round(100/btts, 2)   
    
    home_away = [i, j, home_odds, draw_odds, away_odds, ho_dr, 
                 dr_aw, ha_win, btts_odds, not_btts_odds]
    home_away_df = pd.DataFrame(home_away)
    home_away_trans = home_away_df.transpose()
    home_away_trans.columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 'No BTTS']  
    
    bund2_prediction = bund2_prediction.append(home_away_trans)



