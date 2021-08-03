# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:26:25 2021

@author: peter
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool

def calc_means(champ_param_dict, homeTeam, awayTeam):
    return [np.exp(champ_param_dict['attack_'+homeTeam] + champ_param_dict['defence_'+awayTeam] + champ_param_dict['home_adv']),
            np.exp(champ_param_dict['defence_'+homeTeam] + champ_param_dict['attack_'+awayTeam])]

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

def dixon_coles_simulate_match(champ_params_dict, homeTeam, awayTeam, max_goals=5):
    team_avgs = calc_means(champ_params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], champ_params_dict['rho']) for away_goals in range(2)]
                                   for home_goals in range(2)])
    output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
    return output_matrix

def solve_parameters_decay(dataset, xi=0.001, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
    teams = np.sort(dataset['Home'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['Away'].unique())
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

    def estimate_paramters(champ_params):
        score_coefs = dict(zip(teams, champ_params[:n_teams]))
        defend_coefs = dict(zip(teams, champ_params[n_teams:(2*n_teams)]))
        rho, gamma = champ_params[-2:]
        log_like = [dc_log_like_decay(row.HomeGoals, row.AwayGoals, score_coefs[row.Home], defend_coefs[row.Home],
                                      score_coefs[row.Away], defend_coefs[row.Away], 
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


""""def champ_match_day(HomeTeam, AwayTeam):
    matrix = dixon_coles_simulate_match(champ_params, HomeTeam, AwayTeam, max_goals=5)
    
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
    return matrix, home_win, draw, away_win, ho_dr, dr_aw, ha_win, home_odds, draw_odds, away_odds"""
    
# This is the start of the code without functions

poland_1220 = pd.DataFrame()
poland_all = pd.read_csv("https://www.football-data.co.uk/new/POL.csv")
poland_all['Date'] = pd.to_datetime(poland_all['Date'],  format='%d/%m/%Y')
poland_all['time_diff'] = (max(poland_all['Date']) - poland_all['Date']).dt.days
poland_1220 = poland_all[['Date', 'Home','Away', 'HG','AG', 'Res', 'time_diff']]
poland_1220 = poland_1220.rename(columns={'HG': 'HomeGoals', 'AG': 'AwayGoals'})
poland_1220 = poland_1220.dropna(how='all')
 

poland_1220 = poland_1220[poland_1220['Date'] < '2021-07-01']
    
poland_params = solve_parameters_decay(poland_1220, xi = 0.00325)

poland_prediction = pd.DataFrame(columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 'No BTTS'])

HomeTeam = ['Cracovia', 'Jagiellonia', 'Lech Poznan', 'Legia', 'Piast Gliwice',
            'Pogon Szczecin', 'Slask Wroclaw', 'Wisla Plock']
AwayTeam = ['Warta Poznan', 'Lechia Gdansk', 'Gornik Z.', 'Podbeskidzie',
            'Wisla', 'Rakow', 'Stal Mielec', 'Zaglebie']

for i, j in zip(HomeTeam, AwayTeam):
    # Gives odds on all the scores up to 10 goals for each team, probably overkill
    # Creates a matrix with all of the results
    matrix = dixon_coles_simulate_match(poland_params, i, j, max_goals=5)
    
    # Change the matrix into a DataFrame
    matrix_df = pd.DataFrame(matrix)
    
    # Multiply this by 100 to make the maths easier
    matrix_df = matrix_df * 100
    
    # Sums the triangle of the matrix where the home team would win
    home_win = np.sum(np.tril(matrix, -1))
    
    # Sums the diagonal which will indicate a draw
    draw = np.sum(np.diag(matrix))
    
    # Sums the triangle of the matrix where the away team would win
    away_win = np.sum(np.triu(matrix, 1))
    
    # Find the odds of the home team, draw and away team. Round this to 2dp
    home_odds = round(1/home_win, 2)
    draw_odds = round(1/draw, 2)
    away_odds = round(1/away_win, 2)
    
    # Find the odds for 1X, X2, 12. Round this to 2dp
    ho_dr = round(1/(home_win + draw), 2)
    dr_aw = round(1/(draw + away_win), 2)
    ha_win = round(1/(home_win + away_win), 2)
    
    # Add a row and column which sums the amount of goals for each team
    matrix_df.loc['Total', :] = matrix_df.sum(axis = 0)
    matrix_df.loc[:, 'Total'] = matrix_df.sum(axis = 1)
  
    # Add up the scores for where the both teams do not score
    not_btts = (matrix_df.iloc[-1, 0] + matrix_df.iloc[0, -1] - matrix_df.iloc[0, 0])
    
    # 100 - not_btts to find out both teams to score
    btts = 100 - not_btts
    
    # Find the odds for btts and not_btts
    not_btts_odds = round(100/not_btts, 2)
    btts_odds = round(100/btts, 2)   
    
    # Add up the parts of the matrix where there are under 2.5 goals
    U2_5G = (matrix_df.iloc[0, 0] + matrix_df.iloc[0, 1]
             + matrix_df.iloc[0, 2] + matrix_df.iloc[1, 0]
             + matrix_df.iloc[2, 0] + matrix_df.iloc[1, 1])
    
    # 100 - U2_5G to find O2_5G
    O2_5G = 100 - U2_5G
  
    # Calculate the odds for under and over 2.5 goals, rounded to 2dp
    U2_5G_odds = round(100/U2_5G, 2)
    O2_5G_odds = round(100/O2_5G, 2) 
    
    #Looking at Asian handicaps.
    
    # Looking at -1.5 asian handicap. Take the odds of winning and then minus
    # all the odds where the scores would be only 1 goal difference to away team
    away_minus_1_5 = (np.sum(np.triu(matrix, 1)) - matrix[0, 1] -
                matrix[1, 2] - matrix[2, 3] - 
                matrix[3, 4] - matrix[4, 5] -
                matrix[5, 6] - matrix[6, 7] -
                matrix[7, 8] - matrix[8, 9] - 
                matrix[9, 10])
    
    # Looking at -1.5 asian handicap. Take the odds of winning and then minus
    # all the odds where the scores would be only 1 goal difference to home team
    home_minus_1_5 = (np.sum(np.tril(matrix, -1)) - matrix[1, 0] -
                matrix[2, 1] - matrix[3, 2] - 
                matrix[4, 3] - matrix[5, 4] -
                matrix[6, 5] - matrix[7, 6] -
                matrix[8, 7] - matrix[9, 8] - 
                matrix[10, 9])
    
    # Look at the home team +1.5 goals
    home_plus_1_5 = 1 - away_minus_1_5   
    
    # Look at the away team +1.5 goals
    away_plus_1_5 = 1- home_minus_1_5

    # Work out the odds for all of the above.
    home_plus_1_5_odds = round(1/home_plus_1_5, 2)
    home_minus_1_5_odds = round(1/home_minus_1_5, 2)
    away_plus_1_5_odds = round(1/away_plus_1_5, 2)
    away_minus_1_5_odds = round(1/away_minus_1_5, 2)

    # Create a list for each home team, away team and all the calculations above
    home_away = [i, j, home_odds, draw_odds, away_odds, ho_dr, 
                 dr_aw, ha_win, btts_odds, not_btts_odds, 
                 O2_5G_odds, U2_5G_odds, home_plus_1_5_odds, 
                 home_minus_1_5_odds, away_plus_1_5_odds]
    
    # Turn the above into a DataFrame
    home_away_df = pd.DataFrame(home_away)
    
    # Transpose the above DataFrame
    home_away_trans = home_away_df.transpose()
    
    # Name the columns
    home_away_trans.columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 
                                      'No BTTS', 'Over 2.5G', 'Under 2.5G',
                                      'Home +1.5G', 'Home -1.5G', 'Away +1.5G',
                                      'Away -1.5G']
   
    # Append the above onto the epl_prediction for the two teams ran above
    poland_prediction = poland_prediction.append(home_away_trans)




