import random
import joblib
import pandas as pd
import numpy as np
import time
import dataframe_image as dfi
import matplotlib.pyplot as plt

def fixture_update(fixtures_table):

    fixtures_table = fixtures_table.dropna(subset=['Home'])
    fixtures_table = fixtures_table[~fixtures_table["Score"].str.contains(r"\(", na=False)]
    fixtures = []

    for _, row in fixtures_table.iterrows():

        home_team = row['Home']
        away_team = row['Away']

        if pd.isna(row['Score']):
            result = -1
        else:
            home_score = int(row['Score'][0])
            away_score = int(row['Score'][2])
            result = float(np.where(home_score > away_score, 1, np.where(home_score == away_score, 0.5, 0)))

        fixtures.append((home_team, away_team, result))

    return fixtures

def calculate_expected_score(elo_a, elo_b):
    """Calculate the expected score for Team A against Team B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, K, result_a):
    """
    Update Elo ratings for two teams.
    - elo_a, elo_b: Current Elo ratings of teams A and B
    - result_a: Result for Team A (1 = win, 0 = loss, 0.5 = draw)
    """
    expected_a = calculate_expected_score(elo_a, elo_b)

    
    elo_a_new = elo_a + K * ((result_a - expected_a))
    elo_b_new = elo_b + K * (((1 - result_a) - (1 - expected_a)))
    return elo_a_new, elo_b_new

def simulate_match_elo(team_ratings, team_a, team_b, known_result, K):

    elo_a = team_ratings[team_a]
    elo_b = team_ratings[team_b]

    if known_result == -1:
        """Simulate a match between two teams."""
        expected_a = calculate_expected_score(elo_a, elo_b)
        
        # Simulate result based on expected score
        random_value = random.random()
        if random_value < expected_a:
            result_a = 1  # Team A wins
        elif random_value < expected_a + (1 - expected_a) / 2:
            result_a = 0.5  # Draw
        else:
            result_a = 0  # Team B wins
    else:
        result_a = known_result
    
    # Update Elo ratings
    new_elo_a, new_elo_b = update_elo(elo_a, elo_b, K, result_a)
    team_ratings[team_a] = round(new_elo_a)
    team_ratings[team_b] = round(new_elo_b)

def season_resulting(old_ratings, fixtures_table, K):

    relegation_ratings = sum(sorted(list(old_ratings.values()))[:3]) / 3
    old_teams = list(old_ratings.keys())
    current_teams = list(fixtures_table['Home'].unique())
    new_ratings = {}
    
    for team in current_teams:

        if team in old_teams:
            value = old_ratings[team]
        else:
            value = relegation_ratings

        new_ratings[f'{team}'] = value + ((random.random()-0.5)*400)

    fixtures = fixture_update(fixtures_table)

    for fixture in fixtures:
        simulate_match_elo(new_ratings, fixture[0], fixture[1], fixture[2], K)
    
    return new_ratings

def fixture_tables(league_dict, league):

    fixtures_tables_dict = {}

    for year in range(20, 25):

        time.sleep(5)
        season = '20' + str(year) + '-20' + str(year+1)
        url = 'https://fbref.com/en/comps/' + str(league_dict[league]) + '/' + season + '/schedule/' + season + '-' + league + '-Scores-and-Fixtures'
        tables = pd.read_html(url)
        fixtures_table = tables[0]
        fixtures_table = fixtures_table.dropna(subset=['Home'])
        fixtures_tables_dict[f'fixtures_table{str(year+1)}'] = fixtures_table

    return fixtures_tables_dict


def elo_settler(fixtures_tables_dict, K):

    ratings = {}

    for team in list(fixtures_tables_dict['fixtures_table21']['Home'].unique()):
        ratings[f"{team}"] = 1000
    
    for year in range(20,24):

        table = 'fixtures_table' + str(year+1)
        fixtures_table = fixtures_tables_dict[table]
        ratings = season_resulting(ratings, fixtures_table, K)

    return ratings

def update_league_table(league_table, team_a, team_b, result_a):
    """Update league table based on match result."""

    if result_a == 1:  # Team A wins
        league_table.loc[team_a, "W"] += 1
        league_table.loc[team_b, "L"] += 1
    elif result_a == 0.5:  # Draw
        league_table.loc[team_a, "D"] += 1
        league_table.loc[team_b, "D"] += 1
    else:  # Team B wins
        league_table.loc[team_b, "W"] += 1
        league_table.loc[team_a, "L"] += 1

def simulate_match(team_ratings, league_table, team_a, team_b, known_result, K):

    elo_a = team_ratings[team_a]
    elo_b = team_ratings[team_b]
    
    if known_result == -1:
        """Simulate a match between two teams."""
        expected_a = calculate_expected_score(elo_a, elo_b)
        
        # Simulate result based on expected score
        random_value = random.random()
        if random_value < expected_a:
            result_a = 1  # Team A wins
        elif random_value < expected_a + (1 - expected_a) / 2:
            result_a = 0.5  # Draw
        else:
            result_a = 0  # Team B wins
    else:
        result_a = known_result
    
    # Update Elo ratings
    new_elo_a, new_elo_b = update_elo(elo_a, elo_b, result_a, K)
    team_ratings[team_a] = round(new_elo_a)
    team_ratings[team_b] = round(new_elo_b)

    update_league_table(league_table, team_a, team_b, result_a)

def season_sim(n, fixtures_tables_dict, K):

    fixtures_table = fixtures_tables_dict['fixtures_table25']
    current_teams = list(fixtures_table['Home'].unique()) 

    points_table = pd.DataFrame({
    "Team": current_teams,
    })

    position_table = pd.DataFrame({
        "Team": current_teams,
    })

    sim_table = pd.DataFrame({
        "Team": current_teams,
    }).sort_values(by=["Team"], ascending=True).reset_index(drop=True)


    for i in range(n):
        team_ratings = elo_settler(fixtures_tables_dict, K)

        relegation_ratings = sum(sorted(list(team_ratings.values()))[:3]) / 3
        old_teams = list(team_ratings.keys())
        new_ratings = {}
        
        for team in current_teams:

            if team in old_teams:
                value = team_ratings[team]
            else:
                value = relegation_ratings

            new_ratings[f'{team}'] = value + ((random.random()-0.5)*400)  

        team_ratings_iteration = new_ratings.copy()

        iteration = str(i+1)

        league_table = pd.DataFrame({
        "Team": new_ratings.keys(),
        "W": 0,
        "D": 0,
        "L": 0,
        "Points": 0
        }).set_index("Team")

        fixtures = fixture_update(fixtures_table)

        for fixture in fixtures:
            simulate_match(team_ratings_iteration, league_table, fixture[0], fixture[1], fixture[2], K)

        league_table['Points'] = (3* league_table['W']) + (1* league_table['D'])
        league_table_iteration = league_table.sort_values(by=["Points", "W"], ascending=False).reset_index()
        league_table_iteration['Position'] = league_table_iteration.index + 1

        points_table = pd.merge(points_table, league_table_iteration[['Team', 'Points']], how = 'left', on = 'Team')
        points_table = points_table.rename(columns={"Points": f"Points_{iteration}"})
        points_table = points_table.sort_values(by=["Team"], ascending=True).reset_index(drop=True)
        
        position_table = pd.merge(position_table, league_table_iteration[['Team', 'Position']], how = 'left', on = 'Team')
        position_table = position_table.rename(columns={"Position": f"Position_{iteration}"})
        position_table = position_table.sort_values(by=["Team"], ascending=True).reset_index(drop=True)
    
        

    sim_table['Position (Mean)'] = round(position_table.iloc[:, 1:].mean(axis=1),2)
    sim_table['Points (Upper)'] = points_table.iloc[:, 1:].apply(lambda row: round(row[row > row.median()].mean(), 2), axis=1)
    sim_table['Points (Mean)'] = round(points_table.iloc[:, 1:].mean(axis=1),2)
    sim_table['Points (Lower)'] = points_table.iloc[:, 1:].apply(lambda row: round(row[row < row.median()].mean(), 2), axis=1)
    

    df = position_table.drop(columns = ['Team'])

    countdf = pd.DataFrame()

    for threshold in range(1,21):
        counts = (df.apply(lambda row: sum(row == threshold), axis=1) / (len(df.columns)))
        counts = pd.DataFrame(list(counts), columns = [str(threshold)])
        countdf = pd.concat([countdf, counts], axis=1)

    countdf = countdf.fillna(0)
    sim_table_exact = pd.concat([sim_table, countdf], axis = 1).sort_values(by=["Position (Mean)", "Points (Mean)"], ascending=[True,False]).reset_index(drop=True)

    sim_table_select = pd.DataFrame()

    sim_table_select['Team'] = sim_table_exact['Team']
    sim_table_select['Position'] = round(sim_table_exact['Position (Mean)'],2)
    sim_table_select['Points'] = round(sim_table_exact['Points (Mean)'],2)
    sim_table_select['Points (Upper)'] = round(sim_table_exact['Points (Upper)'],2)
    sim_table_select['Points (Lower)'] = round(sim_table_exact['Points (Lower)'],2)
    sim_table_select['Champion (%)'] = round(sim_table_exact['1'],4) * 100
    sim_table_select['1 - 4 (%)'] = round(sim_table_exact.iloc[:, 5:9].sum(axis=1),4) * 100
    sim_table_select['5 - 8 (%)'] = round(sim_table_exact.iloc[:, 9:13].sum(axis=1),4) * 100
    sim_table_select['9 - 12 (%)'] = round(sim_table_exact.iloc[:, 13:17].sum(axis=1),4) * 100
    if len(sim_table) == 20:
        sim_table_select['13 - 17 (%)'] = round(sim_table_exact.iloc[:, 17:22].sum(axis=1),4) * 100
        sim_table_select['Relegation Zone (%)'] = round(sim_table_exact.iloc[:, -3:].sum(axis=1),4) * 100
    else:
        sim_table_select['13 - 15 (%)'] = round(sim_table_exact.iloc[:, 17:20].sum(axis=1),4) * 100
        sim_table_select['Relegation Zone (%)'] = round(sim_table_exact.iloc[:, -3:].sum(axis=1),4) * 100
        
    sim_table_select.index = range(1, len(sim_table_select) + 1)

    return sim_table_select

def league_simulator(league_dict, league, n, K):

    fixtures_tables_dict = fixture_tables(league_dict, league)
    
    sim_table = season_sim(n, fixtures_tables_dict, K)

    return sim_table

def save_df_as_image(df, file_path):

    df_copy = df.copy()

    for col in df_copy.select_dtypes(include=['number']).columns:
        df_copy[col] = df_copy[col].map("{:.2f}".format)

    fig, ax = plt.subplots(figsize=(len(df_copy.columns) * 2, len(df_copy) * 0.6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_copy.values,
        colLabels=df_copy.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(df_copy.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()

def overall_sim(league_dict, n, K):

    starttime = time.time()
    top5 = list(league_dict.keys())
    sim_table_dict = {}
    for league in top5:
        looptime = time.time()
        time.sleep(5)

        png = '/home/musamoholo98/ELO-Simulation-Model/output/' + league + '.png'
        
        sim_table = league_simulator(league_dict, league, n, K)
        save_df_as_image(sim_table, png)
        sim_table_dict[f'{league}'] = sim_table
        timetaken = int((time.time() - looptime)/60)
        timeinrun = int((time.time() - starttime)/60)

        message = f'{league} simulated in ~{timetaken} mins ({timeinrun} mins overall)'
        print(message)

    return sim_table_dict

def plot_dataframes(dict):

    num_dfs = len(dict)
    fig, axes = plt.subplots(num_dfs, 1, figsize=(12, 6 * num_dfs))

    if num_dfs == 1:
        axes = [axes]  # Ensure it's iterable if there's only one dataframe


    for ax, (key, df) in zip(axes, dict.items()):
        # Create a copy of the DataFrame and format numeric columns to 2 decimal places
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['number']).columns:
            df_copy[col] = df_copy[col].map("{:.2f}".format)
        
        # Plot the table
        ax.axis('tight')
        ax.axis('off')
        ax.table(
            cellText=df_copy.values,
            colLabels=df_copy.columns,
            rowLabels=df_copy.index,
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2'] * len(df_copy.columns)  # Light gray background for columns
        )
        ax.set_title(key, fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()