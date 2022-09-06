#testing if changes to the attraction scores affect the results of the linear program

from typing import List
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
import os

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, PULP_CBC_CMD
import re
from fuzzywuzzy import process

import pdb

routes = pd.read_csv('outputs/selected_routes.csv')
attractions = pd.read_csv('mr-qap/mr-qap-results.csv').round(2)
refugee_data = pd.read_csv('refugee_data/refugee_conflict_5.20.22.csv')

total_refugees = refugee_data[refugee_data.conflict=='Ukraine'].refugees.sum()

features = ['PPL', 'PPLA', 'PPLA2',' PPLA3', 'PPLA4', 'PPLA5','PPLC','PPLCH','PPLF','PPLG','PPLH','PPLQ','PPLR','PPLS',
            'PPLW','PPLX','RGN']

CITY_FILE = "inputs/UA.txt"
city_df = pd.read_csv(
    CITY_FILE,
    sep="\t",
    header=0,
    names=[
        "geonameid",
        "name",
        "asciiname",
        "alternatenames",
        "latitude",
        "longitude",
        "feature class",
        "feature code",
        "country code",
        "cc2",
        "admin1 code",
        "admin2 code",
        "admin3 code",
        "admin4 code",
        "population",
        "elevation",
        "dem",
        "timezone",
        "modification date",
    ],
    low_memory=False
)

city_df = city_df[city_df['feature code'].isin(features)]

subset_cols = ["name", "latitude", "longitude", "country code", "population"]
city_df = city_df[city_df['country code'] == 'UA'].reset_index()
city_df = city_df[subset_cols]
city_df['lat'] = city_df.latitude.round(0)
city_df['lon'] = city_df.longitude.round(0)

population = pd.DataFrame(city_df.groupby(['lat','lon']).population.sum()).reset_index()

routes['lat'] = routes.conflict_lat.round(0)
routes['lon'] = routes.conflict_lon.round(0)

routes = pd.merge(routes, population, left_on=['lat','lon'], right_on=['lat','lon'])

del(routes['lat'])
del(routes['lon'])


total_affected_population = routes.drop_duplicates(subset=['conflict']).population.sum()
refugee_ratio = total_refugees / total_affected_population

# ## Generate Linear Optimization Model

def adjust_duration_transit(row):
    """
    Hardcoded adjustment to transit (divide duration by 4) as an equivalency measure to driving.
    In that case, a 8 hour train ride is equivalent to a 2 hour car ride.
    """
    if row['mode'] == 'transit':
        row.duration = row.duration / 4
    return row

routes = routes.apply(lambda row: adjust_duration_transit(row), axis=1)
attractions = dict(zip(attractions.country, attractions.predicted_shares))


def run_sensitivity(attractions):

    def get_population(conflict):
        population = routes[routes.conflict==conflict].iloc[0].population
        return population


    reg = re.compile(r'[\W]')
    def strip_text(text):
        return re.sub(reg, '', text).lower()


    conflicts = routes.conflict.unique()
    crossings = routes.crossing.unique()
    countries = routes.crossing_country.unique()

    model = LpProblem(name="refugee-routing", sense=LpMinimize)

    variables = {}
    variables_lookup = {}

    for conf in conflicts:
        variables[conf] = []
        
    for kk, vv in routes.iterrows():
        conf = strip_text(vv.conflict)
        cross = strip_text(vv.crossing)
        country = strip_text(vv.crossing_country)
        mode = vv['mode']
        v = f"{conf}__{cross}__{country}__{mode}"
        x = LpVariable(name=v, lowBound=0, cat='Continuous')
        variables[vv.conflict].append(x)
        
        variables_lookup[v] = dict(conflict=vv.conflict,
                                crossing=vv.crossing,
                                country=vv.crossing_country,
                                mode=mode,
                                duration=vv.duration)

    vars_by_country = {}
    for country in countries:
        vars_by_country[country] = []

    for kk, vv in variables.items():
        for i in vv:
            country = variables_lookup[i.name]['country']
            vars_by_country[country].append(i)

    for conf in conflicts:
        pop = get_population(conf)
        model += (lpSum(variables[conf]) == round(refugee_ratio * pop,0), 
                f"{conf}_refugee_constraint") 

    for c in countries[:6]:
        country_vars = vars_by_country[c]
        attraction = round(attractions[c],2)
        
        model += (lpSum(country_vars)/total_refugees == attraction,
                f"{c}_attraction")

    obj_func_array = []
    for kk, vv in variables.items():
        for i in vv:
            refugees = i
            duration = variables_lookup[i.name]['duration']
            obj_func_array.append(refugees*duration)

    model += lpSum(obj_func_array)

    # pdb.set_trace()
    status = model.solve(PULP_CBC_CMD(msg=0))

    refugee_counts = {}
    for c in countries:
        refugee_counts[c] = 0

    for var in model.variables():
        for c in countries:
            if c.split(' ')[0].lower() in var.name.lower():
                refugee_counts[c] += var.value()


    results = []
    for var in model.variables():
        results.append((var, var.value()))
    results = pd.DataFrame(results, columns=['variable','value'])    



    results['conflict'] = results.variable.apply(lambda x: variables_lookup[x.name]['conflict'])
    results['crossing'] = results.variable.apply(lambda x: variables_lookup[x.name]['crossing'])
    results['country'] = results.variable.apply(lambda x: variables_lookup[x.name]['country'])
    results['mode'] = results.variable.apply(lambda x: variables_lookup[x.name]['mode'])
 
    return results





##################### Stuff for sensitivity test #####################
output_dir = 'sensitivity_analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

attraction_keys = list(attractions.keys())
model_variables = None
def initialize_results():
    results = {}
    for i in attraction_keys:
        results[i] = {}
        for d in model_variables:
            results[i][d] = [[], []] #list of points
    return results

def get_sample_proportions(n):
    props = []
    prop_remaining = 1
    while len(props) < n - 1:
        prop = np.random.uniform(0, prop_remaining)
        props.append(prop)
        prop_remaining -= prop
    props.append(prop_remaining)
    props = np.array(props)

    #shuffle
    np.random.shuffle(props)

    return props

def get_variables():
    """run a single experiment, and pull the list of model variables"""
    attractions = get_sample_proportions(len(attraction_keys))
    attractions = dict(zip(attraction_keys, attractions))
    results = run_sensitivity(attractions)
    return [v.name for v in results.variable.tolist()]


# sensitivity_results = None
runs = 10 #100 #number of runs to average results over
divs = 20 #number of divisions to check attractions at 


country_matrices = {}
variables = get_variables()

for i, country in enumerate(tqdm(attraction_keys, desc='sensitivity over countries')):
    country_vars = [v for v in variables if v.lower().split('__')[-2] == country.lower().replace(' ', '')]
    xnew = np.linspace(0, 1, divs+1)
    mat = pd.DataFrame(np.zeros((len(country_vars), divs+1)), index=country_vars, columns=xnew)
    mat.columns.name = country
    country_matrices[country] = mat

    for xi in tqdm(xnew, leave=False, desc=f'{country} attraction'):
        for run in trange(runs, leave=False, desc='average over runs'):
            attraction_values = get_sample_proportions(len(attraction_keys)-1)
            attraction_values *= (1-xi)
            attraction_values = np.insert(attraction_values, i, xi)
            attractions = dict(zip(attraction_keys, attraction_values))
            results = run_sensitivity(attractions)
            
            #collect results that are for this country
            results = results[results.country == country].copy()
            results['variable'] = results.variable.apply(lambda x: x.name)            

            #add results to matrix
            for var in results.variable.tolist():
                country_matrices[country].loc[var, xi] += results[results.variable == var].value.values[0] / runs
            

    #save results to csv
    country_matrices[country].to_csv(os.path.join(output_dir, f'{country}_sensitivity.csv'))



#plot each of the mats
for country, mat in country_matrices.items():
    fig, ax = plt.subplots()
    sns.heatmap(mat, ax=ax)
    ax.set_title(country)
    plt.savefig(os.path.join(output_dir, f'{country}_sensitivity.png'))
    plt.close(fig)       



#plot the results
# for country, data in tqdm(sensitivity_results.items(), desc='saving plots for each country'):
#     for var, values in tqdm(data.items(), leave=False, desc='saving plots for each variable'):
#         plt.scatter(values[0], values[1], alpha=0.1)
#         plt.title(f"{country} | {var}")
#         plt.xlabel('attraction')
#         plt.ylabel('refugees')
#         #save the plot
#         plt.savefig(f"plots/{country}__{var}.png")
#         plt.close()
