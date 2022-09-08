#testing if changes to the attraction scores affect the results of the linear program

from typing import List
from tqdm import trange, tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib import pyplot as plt, ticker


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

original_attractions = attractions.copy() #for plotting compared to original
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

def get_country_results(xi):
    #parallelized version over all runs
    attraction_values = get_sample_proportions(len(attraction_keys)-1)
    attraction_values *= (1-xi)
    attraction_values = np.insert(attraction_values, i, xi)
    attractions = dict(zip(attraction_keys, attraction_values))
    results = run_sensitivity(attractions)
    results['variable'] = results.variable.apply(lambda x: x.name) #replace LpVariable with string
    return results

def average_results(country_result):
    return pd.concat(country_result)[['variable', 'value']].groupby('variable').mean()

# sensitivity_results = None
runs = 200 #number of runs to average results over
divs = 20  #number of divisions to check attractions at 



for i, country in enumerate(tqdm(attraction_keys, desc='sensitivity over countries')):
    xnew = np.linspace(0, 1, divs+1) #what attraction percentages to check

    def index_generator():
        for xi in xnew: #attraction percent
            for run in range(runs): #runs to average over
                yield xi

    #run big parallellized loop for all runs + xi values
    country_results = process_map(get_country_results, [*index_generator()], chunksize=50, leave=False, desc=f'Average {country} runs')

    #reshape country results to be runs * divs+1
    country_results = [country_results[i*runs:(i+1)*runs] for i in range(len(xnew))]

    #average over runs
    country_results = [average_results(r) for r in country_results]

    #concatenate results along rows, with columns as xi, removing old 'value' from column names
    country_results = pd.concat(country_results, axis=1, keys=xnew)
    country_results.columns = country_results.columns.droplevel(1)
    
    #remove all non-country variables
    country_results['country'] = country_results.index.map(lambda x: x.lower().split('__')[-2])
    country_results = country_results[country_results.country == country.lower().replace(' ', '')]
    country_results = country_results.drop('country', axis=1)

    
    ######### preprocess + sort the matrix for plotting #########
    
    mat = country_results.copy()
    #sort first by mode, and then by mean value across all columns
    mean = mat.mean(axis=1)
    first = (mat.values > 0).argmax(axis=1)
    first[mat.values.max(axis=1) <= 0] = first.max() + 1
    mat['mean'] = mean
    mat['first'] = first
    mat['mode'] = mat.index.map(lambda x: x.split('__')[-1])
    mat = mat.sort_values(['mode', 'first', 'mean'], ascending=[False, True, False], kind='mergesort')
    
    #get the index that mode switches from 'driving' to 'transit'
    mode_switch_var = mat[mat['mode'] == 'driving'].index[0]
    mode_switch_idx = mat.index.get_loc(mode_switch_var)
    mode_switch_val = mode_switch_idx / len(mat) #normalize to 0-1

    #drop the sorting columns
    mat = mat.drop(['mode', 'mean', 'first'], axis=1)

    #replace negative values with 0
    mat[mat < 0] = 0
    

    ######### actual plotting here #########
    data = mat.values
    
    #set font sizes for titles and axis labels
    plt.rc('axes', labelsize=14, titlesize=16)

    #plot the data
    plt.imshow(data, cmap='inferno', aspect='auto', origin='lower')

    #colorbar scaled to millions
    plt.colorbar(label='refugees (millions)', format=ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1e6)))
    num_xticks = 11
    xticks = np.linspace(0, divs+1, num_xticks)-0.5
    xlabels = [f'{x*100:.0f}' for x in np.linspace(0, 1, num_xticks)]
    plt.xticks(xticks, xlabels, rotation='horizontal', ha='center')
    plt.yticks([])
    plt.title('Route Utilization')
    plt.xlabel('Attraction %')
    

    #add the mode switch line with centering
    plt.axhline(mode_switch_idx-0.5, color='white', linestyle='--')
    
    #plot the hierarchical y-axis label (driving vs transit)
    ax2 = plt.gca().twinx()
    ax2.spines["left"].set_position(("axes", -0.025))
    ax2.tick_params('both', length=0, width=0, which='minor')
    ax2.tick_params('both', direction='in', which='major')
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_yticks([0.0, mode_switch_val, 1.0])
    ax2.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(ticker.FixedLocator([mode_switch_val/2, (mode_switch_val+1)/2]))
    ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(['Transit', 'Driving']))
    plt.setp(ax2.yaxis.get_minorticklabels(), rotation=90, va="center")
    plt.ylabel('Route') #ylabel here so that it is relative to the parasite axes

    #add the original attraction as a vertical line + label
    original_attraction = original_attractions[country] * divs
    plt.axvline(original_attraction, color='r', linestyle='--')
    plt.text(original_attraction, 0.5, 'MR-QAP\nAttraction', rotation=90, va='center', ha='center', color='w')


    # plt.show()
    plt.savefig(os.path.join(output_dir, f'{country}_sensitivity.png')); plt.close()     


    #plot the log of the data
    plt.imshow(np.log(data + 1e-8), cmap='inferno', aspect='auto', origin='lower')
    plt.xticks(xticks, xlabels, rotation='horizontal', ha='center')
    plt.yticks([])
    plt.title('Log Route Utilization')
    plt.xlabel('Attraction %')
    #add the mode switch line with centering
    plt.axhline(mode_switch_idx-0.5, color='white', linestyle='--')
    plt.axvline(original_attraction, color='r', linestyle='--')
    plt.colorbar(label='log(refugees)', format=ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x)))


    # plt.show()
    plt.savefig(os.path.join(output_dir, f'{country}_sensitivity_log.png')); plt.close()



    ########### save results to csv ###########
    #invert the order of the rows (since the plots are inverted)
    mat = mat.iloc[::-1]
    mat.to_csv(os.path.join(output_dir, f'{country}_sensitivity.csv'))