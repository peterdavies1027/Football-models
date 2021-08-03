# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:39:36 2021

@author: peter
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool

def calc_means(spain_param_dict, homeTeam, awayTeam):
    return [np.exp(spain_param_dict['attack_'+homeTeam] + spain_param_dict['defence_'+awayTeam] + spain_param_dict['home_adv']),
            np.exp(spain_param_dict['defence_'+homeTeam] + spain_param_dict['attack_'+awayTeam])]

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

def dixon_coles_simulate_match(spain_params_dict, homeTeam, awayTeam, max_goals=5):
    team_avgs = calc_means(spain_params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], spain_params_dict['rho']) for away_goals in range(2)]
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

    def estimate_paramters(spain_params):
        score_coefs = dict(zip(teams, spain_params[:n_teams]))
        defend_coefs = dict(zip(teams, spain_params[n_teams:(2*n_teams)]))
        rho, gamma = spain_params[-2:]
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

def solve_parameters_decay_YC(dataset, xi=0.001, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
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
  
    
# This is the start of the code without functions

# Create a blank DataFrame for all fixtures in the EPL  
bund2_all = pd.DataFrame()

# Get the data from football-data for the seasons 2017-2021, this will need to 
# be changed each year for the newest season
for year in range(18,21):
    
    # Concatenate all of them together into 1 DataFrame
    bund2_all = pd.concat((bund2_all, pd.read_csv("http://www.football-data.co.uk/mmz4281/{}{}/D2.csv".format(year, year+1, sort=True))))

# Ensure that the date is ina  sensible format, day/month/year
bund2_all['Date'] = pd.to_datetime(bund2_all['Date'],  format='%d/%m/%Y')

# Create a variable for time difference, this will be the number of days
# from today. This variable is a factor that would be used to give more 
# weight to the latest matches.
bund2_all['time_diff'] = (max(bund2_all['Date']) - bund2_all['Date']).dt.days

######

# We are only interested in the date, home team, away team, goals, results
# and the time difference
bund2_1720 = bund2_all[['Date', 'HomeTeam','AwayTeam','FTHG','FTAG', 'FTR', 'time_diff']]

# Rename some columns to sensible names
bund2_1720 = bund2_1720.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

# Not interested in anything with na in this dataset
bund2_1720 = bund2_1720.dropna(how='all')

######

# Same dataset as above however we are interested in yellow cards rather than goals
bund2_YC_1720 = bund2_all[['HomeTeam', 'AwayTeam', 'HY', 'AY', 'FTR', 'time_diff']]

# Rename some columns so they are of more use
bund2_YC_1720 = bund2_YC_1720.rename(columns={'HY': 'HomeYellowC', 'AY': 'AwayYellowC'})

# Not interested in anything with na in this dataset
bund2_YC_1720 = bund2_YC_1720.dropna(how='all')

######

# Creates variables for attack and defence for each team, giving more weight
# to the most rececnt results. Can change the xi value however this seems 
# the best value at the moment. Something to look in to at a later date
bund2_params = solve_parameters_decay(bund2_1720, xi = 0.00325)

# Creates variables for ytellow cards for each team
bund2_params_YC = solve_parameters_decay_YC(bund2_YC_1720, xi = 0.00325)

# Create a DataFrame with all the values we are interested in
bund2_prediction = pd.DataFrame(columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 'No BTTS',
                                      'Over 2.5G', 'Under 2.5G', 'Home +1.5G', 'Home -1.5G', 'Away +1.5G',
                                      'Away -1.5G', 'Home YC win', 'Draw YC', 
                                      'Away YC win', 'Over 2.5YC', 'Under 2.5YC'])
                                      #"""'Home corner win', 'Draw corner', 'Away corner win',
                                      #'Over 9.5 corners', 'Under 9.5 corners'"""])

# List of home teams in the fixtures we are interested in
HomeTeam = ['Arsenal', 'Aston Villa', 'Fulham', 'Leeds', 'Leicester', 'Liverpool',
            'Man City', 'Sheffield United', 'West Ham', 'Wolves']

# List of away teams in the fixtures we are intrested in.
# WARNING this has to be in the same order as above.
AwayTeam = ['Brighton', 'Chelsea', 'Newcastle', 'West Brom','Tottenham', 'Crystal Palace',
            'Everton', 'Burnley', 'Southampton', 'Man United']
   
# This simulates matches between the HomeTeam and AwayTeam in the lists above 
for i, j in zip(HomeTeam, AwayTeam):
    # Gives odds on all the scores up to 10 goals for each team, probably overkill
    # Creates a matrix with all of the results
    matrix = dixon_coles_simulate_match(bund2_params, i, j, max_goals=5)
    
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

    # Same as above but for yellow cards
    matrix_YC = dixon_coles_simulate_match(bund2_params_YC, i, j, max_goals=10)
    
    # Turn the results into a DataFrame and multiply by 100
    matrix_YC_df = pd.DataFrame(matrix_YC)
    matrix_YC_df = matrix_YC_df * 100
    
    # Sum the triangle of the matrix where the home team has more yellow cards
    home_win_YC = np.sum(np.tril(matrix_YC, -1))
    
    # Sum the diagonol where the home and away team have the same cards
    draw_YC = np.sum(np.diag(matrix_YC))
    
    # Sum the triangle of the matrix where the away team has more yellow cards
    away_win_YC = np.sum(np.triu(matrix_YC, 1))
    
    # Find the odds for all the above, rounded to 2dp.
    home_YC_odds = round(1/home_win_YC, 2)
    draw_YC_odds = round(1/draw_YC, 2)
    away_YC_odds = round(1/away_win_YC, 2)
    
    # Sum all of the columns and rows to find the totals for each team
    matrix_YC_df.loc['Total', :] = matrix_YC_df.sum(axis = 0)
    matrix_YC_df.loc[:, 'Total'] = matrix_YC_df.sum(axis = 1)
    
    # Add up the parts where there are less than 2.5 cards
    U2_5YC = (matrix_YC_df.iloc[0, 0] + matrix_YC_df.iloc[0, 1]
             + matrix_YC_df.iloc[0, 2] + matrix_YC_df.iloc[1, 0]
             + matrix_YC_df.iloc[2, 0] + matrix_YC_df.iloc[1, 1])
    
    # 100 - U2.5C to find over 2.5 cards
    O2_5YC = 100 - U2_5YC
    
    # Calculate the odds for each, to 2 dp
    U2_5YC_odds = round(100/U2_5YC, 2)
    O2_5YC_odds = round(100/O2_5YC, 2)   
    
    
    # Create a list for each home team, away team and all the calculations above
    home_away = [i, j, home_odds, draw_odds, away_odds, ho_dr, 
                 dr_aw, ha_win, btts_odds, not_btts_odds, 
                 O2_5G_odds, U2_5G_odds, home_plus_1_5_odds, 
                 home_minus_1_5_odds, away_plus_1_5_odds, 
                 away_minus_1_5_odds, home_YC_odds, draw_YC_odds, 
                 away_YC_odds, O2_5YC_odds, U2_5YC_odds]
    
    # Turn the above into a DataFrame
    home_away_df = pd.DataFrame(home_away)
    
    # Transpose the above DataFrame
    home_away_trans = home_away_df.transpose()
    
    # Name the columns
    home_away_trans.columns = ['HomeTeam', 'AwayTeam', 'Home win', 'Draw', 
                                      'Away win', '1X', 'X2', '12', 'BTTS', 
                                      'No BTTS', 'Over 2.5G', 'Under 2.5G',
                                      'Home +1.5G', 'Home -1.5G', 'Away +1.5G',
                                      'Away -1.5G', 'Home YC win', 'Draw YC', 
                                      'Away YC win','Over 2.5YC', 'Under 2.5YC']
   
    # Append the above onto the epl_prediction for the two teams ran above
    bund2_prediction = bund2_prediction.append(home_away_trans)

