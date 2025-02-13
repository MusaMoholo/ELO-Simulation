This project is designed to predict the final league table for each of Europe’s top 5 football leagues. The project uses the Elo rating system, which is famous for its use in chess, to simulate match results and produce accurate predictions. 
Each league is simulated from five seasons ago, incorporating all match results from those seasons and the current season to establish realistic Elo ratings for every team. 
By the current game week, these ratings are used to simulate the remainder of the season multiple times, ultimately predicting the final standings.

The simulation is run for Europe’s top 5 leagues: the English Premier League (EPL), La Liga (Spain), Serie A (Italy), Bundesliga (Germany), and Ligue 1 (France). 
For each league, the simulation follows a Monte Carlo-based approach, running through the remaining fixtures many times to account for randomness and variability.
After these simulations, a probability table is generated, showing the likelihood of each team finishing in different positions. 
This table provides detailed insights, such as the percentage chance of winning the league, finishing in the top four, or being relegated.

By combining historical data with the Elo system, ELO-Simulation offers a robust framework for forecasting football league outcomes.
It enables users to understand team performance trends and their probabilities of success in the current season based on past and current match data. 
The final output is a comprehensive probability table that reflects the most likely outcomes across thousands of simulated scenarios.
