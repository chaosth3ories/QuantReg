import pandas as pd
import statsmodels.formula.api as smf

dune_master_path = 'DuneMaster.csv'
db_master_path = 'DBMaster.csv'

# Dateneinlesen
dune_master_df = pd.read_csv(dune_master_path)
db_master_df = pd.read_csv(db_master_path)

# Yolo
dune_total_wagered = dune_master_df.groupby('depositor')['deposit_usd'].sum().reset_index()
dune_total_wagered.rename(columns={'depositor': 'player', 'deposit_usd': 'total_wagered'}, inplace=True)
dune_total_wagered['complexity'] = 0

# DB
db_total_wagered = db_master_df.groupby('player')['_wagerAmount'].sum().reset_index()
db_total_wagered.rename(columns={'_wagerAmount': 'total_wagered'}, inplace=True)
db_total_wagered['complexity'] = 1

# Combine Data
combined_wagered_df = pd.concat([dune_total_wagered, db_total_wagered])

# QuantReg
quantiles = [0.25, 0.5, 0.75]

for q in quantiles:
    model = smf.quantreg('total_wagered ~ complexity', combined_wagered_df)
    result = model.fit(q=q)
    print(f'Results for Quantile: {q}')
    print(result.summary())
    print("\n-----------------------------------------------------\n")
