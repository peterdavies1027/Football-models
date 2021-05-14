# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:00:24 2021

@author: peter
"""

#Snippets


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

