import pandas as pd
import statsmodels.formula.api as smf


dune_master_df = pd.read_csv('DuneMaster.csv')
db_master_df = pd.read_csv('DBMaster.csv')

# Amount of Games for Yolo
dune_games_played = dune_master_df.groupby('depositor').size().reset_index(name='games_played')
dune_games_played['complexity'] = 0  # Uncomplex

# Amount of Games DB
db_games_played = db_master_df.groupby('player').size().reset_index(name='games_played')
db_games_played['complexity'] = 1  # Complex

# Formating
dune_games_played.rename(columns={'depositor': 'player'}, inplace=True)

# Date Frames
combined_games_played = pd.concat([dune_games_played, db_games_played], axis=0)

# QuantReg
quantiles = [0.25, 0.5, 0.75]

for q in quantiles:
    model = smf.quantreg('games_played ~ complexity', combined_games_played)
    result = model.fit(q=q)
    print(f'Results for Quantile: {q}')
    print(result.summary())
    print("\n-----------------------------------------------------\n")
