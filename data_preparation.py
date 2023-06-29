import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
import pycountry
warnings.filterwarnings('ignore')

#%%
#all_df = pd.read_csv("data/ess_9.csv")
all_df = pd.read_csv("data/ess_9.csv", low_memory=False)
columns = ['region', 'cntry', 'lrscale', 'ppltrst', 'pplfair', 'pplhlp', 'stfeco']
numericals = ['lrscale', 'ppltrst', 'pplfair', 'pplhlp', 'stfeco']
df = all_df[columns]

df['lrscale'] = df['lrscale'].replace({r"Don\'t know": np.nan,
                                                           "Don't know": np.nan,
                                                           'Refusal': np.nan,
                                                           'Left': 0,
                                                           'Right': 10,
                                                           'No answer': np.nan,
                                                           })

df['ppltrst'] = df['ppltrst'].replace({r"You can't be too careful": 0,
                                       "Most people can be trusted": 10,
                                       r"Don't know": np.nan,
                                       'No answer': np.nan,
                                       'Refusal': np.nan
                                       })

df['pplfair'] = df['pplfair'].replace({r"Most people try to take advantage of me": 0,
                                       "Most people try to be fair": 10,
                                       r"Don't know": np.nan,
                                       'No answer': np.nan,
                                       'Refusal': np.nan
                                       })

df['pplhlp'] = df['pplhlp'].replace({r"People mostly look out for themselves": 0,
                                     "People mostly try to be helpful": 10,
                                     r"Don't know": np.nan,
                                     'No answer': np.nan,
                                     'Refusal': np.nan
                                     })

df['stfeco'] = df['stfeco'].replace({r"Extremely dissatisfied": 0,
                                                     "Extremely satisfied": 10,
                                                     r"Don't know": np.nan,
                                                     'No answer': np.nan,
                                                     'Refusal': np.nan
                                                     })

for col in numericals:
    df[col] = df[col].astype(float)

df['trust'] = df[['ppltrst', 'pplfair', 'pplhlp']].mean(axis=1)
df.drop(['ppltrst', 'pplfair', 'pplhlp'], axis=1, inplace=True)
df = df.dropna()
print('done')


#%%
cntr = df.groupby('cntry').mean(['lrscale', 'trust', 'stfeco'])
cntr = cntr.reset_index()

#%%
iso3_to_iso2 = {c.alpha_3: c.alpha_2 for c in pycountry.countries}
pycntr = px.data.gapminder().query("year==2020")
pycntr["iso_alpha2"] = pycntr["iso_alpha"].map(iso3_to_iso2)
cntr = cntr.merge(pycntr, left_on="cntry", right_on="iso_alpha2")

#%%
fig = px.scatter(
    cntr,
    x="lrscale",
    y="stfeco",
    hover_name="cntry",
    hover_data=[["lrscale", "stfeco"]],
)
fig.update_traces(marker_color="rgba(0,0,0,0)")

#%%
for i, row in df.iterrows():
    print(row["cntry"])


#%%
minDim = df[["lrscale"]].max().idxmax()
maxi = df[minDim].max()
for i, row in df.iterrows():
    country_iso = row["cntr"]
    fig.add_layout_image(
        dict(
            source=f"https://raw.githubusercontent.com/matahombres/CSS-Country-Flags-Rounded/master/flags/{country_iso}.png",
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            x=row["stfeco"],
            y=row["gdpPercap"],
            sizex=np.sqrt(row["pop"] / df["pop"].max()) * maxi * 0.15 + maxi * 0.03,
            sizey=np.sqrt(row["pop"] / df["pop"].max()) * maxi * 0.15+ maxi * 0.03,
            sizing="contain",
            opacity=0.8,
            layer="above"
        )
    )

fig.update_layout(height=600, width=1000, plot_bgcolor="#dfdfdf", yaxis_range=[-5e3, 55e3])

fig.show()

#%% DF TRUST



df_trust['ppltrst'] = df_trust['ppltrst'].astype(float)
df_trust['pplfair'] = df_trust['pplfair'].astype(float)
df_trust['pplhlp'] = df_trust['pplhlp'].astype(float)

df_trust = df_trust.dropna()
df_trust['trust'] = df_trust[['ppltrst', 'pplfair', 'pplhlp']].mean(axis=1)
df_trust = df_trust[['region', 'trust']]
df_trust = df_trust.groupby("region").mean("trust")

#%% ECONOMY
df_economy = all_df[['region', 'stfeco']]

df_economy['stfeco'] = df_economy['stfeco'].astype(float)
df_economy = df_economy.dropna()
df_economy = df_economy.groupby("region").mean("stfeco")

#%%
df_final = pd.concat([df_politics1, df_trust, df_economy], axis=1)


#%%
fig = plt.figure(figsize=(20, 10))
plt.scatter(df_final['lrscale'], df_final['stfeco'])
np.corrcoef(df_final['lrscale'], df_final['stfeco'])


#%%



df = df_sampled[['region', 'polintr']]
df = df.loc[df['polintr']!="Don't know"]
df = df.loc[df['polintr']!="Refusal"]
df['polintr'] = df['polintr'].replace({
    "Not at all interested": 0,
    "Hardly interested": 1,
    "Quite interested": 2,
    "Very interested": 3
})

df['polintr'] = df['polintr'].astype(float)
df = df.groupby("region").mean("polintr")
df.reset_index(inplace=True)







#%%
countries = pd.read_excel('data/countries.xlsx')
countries = countries.iloc[:, 0:2]
df = pd.read_parquet('data_clean/eurobarometer.parquet.gzip')
df = df.loc[df.year >= 2004]





#%%#%%
def variable_by_country(df, column):
    df = df.sample(frac=.01)
    df = df[['country_label', 'year', column]]
    df = df.dropna()
    df = df.groupby(["country_label", "year"]).mean(column)
    df.reset_index(inplace=True)
    df = df.merge(countries, left_on='country_label', right_on="country")
    df = df.groupby(["iso2", "year"]).mean(column)
    df.reset_index(inplace=True)
    return df

#%%
treu = variable_by_country(df, 'treu')
eu_econ = variable_by_country(df, 'econ_eunext_2')

#%%

#%%
df = pd.merge(treu, eu_econ, on=['iso2', 'year'], how='outer')
df.to_parquet("data_clean/eurobarometer_clean.parquet.gzip", compression="gzip")

#%%
