#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:47:03 2020

@author: ashlee
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool

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

def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_cards=10):
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_cards+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_cards, away_cards, team_avgs[0],
                                                   team_avgs[1], params_dict['rho']) for away_cards in range(2)]
                                   for home_cards in range(2)])
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

def epl_match_day(HomeTeam, AwayTeam):
    matrix = dixon_coles_simulate_match(params, HomeTeam, AwayTeam, max_cards=10)
    
    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df * 100
    
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    home_odds = 1/home_win
    draw_odds = 1/draw
    away_odds = 1/away_win
    
    ho_dr = 1/(home_win + draw)
    dr_aw = 1/(draw + away_win)
    ha_win = 1/(home_win + away_win)
    
    matrix_df.loc['Total', :] = matrix_df.sum(axis = 0)
    matrix_df.loc[:, 'Total'] = matrix_df.sum(axis = 1)
    
    not_btts = (matrix_df.iloc[-1, 0] + matrix_df.iloc[0, -1] - matrix_df.iloc[0, 0])
    btts = 100 - not_btts
    
    not_btts_odds = 100/not_btts
    btts_odds = 100/btts    
    
    print(' ')
    print(HomeTeam + str(' vs ') + AwayTeam)
    print(' ')
    print(round(matrix_df, 2))
    print(' ')
    print('Home %:' + str(round((home_win * 100), 2)))
    print('Draw %:' + str(round((draw * 100), 2)))
    print('Away %:' + str(round((away_win * 100), 2)))
    print(' ')
    print('Home odds:' + str(round(home_odds, 2)))
    print('Draw odds:' + str(round(draw_odds, 2)))
    print('Away odds:' + str(round(away_odds, 2)))
    print(' ')
    print('1X odds:' + str(round(ho_dr, 2)))
    print('X2 odds:' + str(round(dr_aw, 2)))
    print('12 odds:' + str(round(ha_win, 2)))
    print(' ')
    print('BTTS:' + str(round(btts_odds, 2)))
    print('Not BTTS:' + str(round(not_btts_odds, 2)))
    print(' ')
    return matrix, home_win, draw, away_win, ho_dr, dr_aw, ha_win, home_odds, draw_odds, away_odds
    
  
    
epl_1720 = pd.DataFrame()
for year in range(17,20):
    epl_1720 = pd.concat((epl_1720, pd.read_csv("http://www.football-data.co.uk/mmz4281/{}{}/E0.csv".format(year, year+1, sort=True))))
epl_1720['Date'] = pd.to_datetime(epl_1720['Date'],  format='%d/%m/%Y')
epl_1720['time_diff'] = (max(epl_1720['Date']) - epl_1720['Date']).dt.days
epl_1720 = epl_1720[['HomeTeam','AwayTeam','HC','AC', 'FTR', 'time_diff']]
epl_1720 = epl_1720.rename(columns={'HC': 'HomeYellowC', 'AC': 'AwayYellowC'})
epl_1720 = epl_1720.dropna(how='all')

      
params = solve_parameters_decay(epl_1720, xi = 0.00325)


ars_wat = epl_match_day('Arsenal', 'Watford')
bur_bri = epl_match_day('Burnley', 'Brighton')
che_wol = epl_match_day('Chelsea', 'Wolves')
cry_tot = epl_match_day('Crystal Palace', 'Tottenham')
eve_bou = epl_match_day('Everton', 'Bournemouth')
lei_manu = epl_match_day('Leicester', 'Man United')
manc_nor = epl_match_day('Man City', 'Norwich')
new_liv = epl_match_day('Newcastle', 'Liverpool')
sou_she = epl_match_day('Southampton', 'Sheffield United')
wes_ast = epl_match_day('West Ham', 'Aston Villa')



