from functools import reduce
import plotly.express as px
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import h3
import numpy as np
from PIL import Image

st.beta_set_page_config(layout="wide")
st.title("Exploring the relation between 311 calls and gentrification in NYC")
st.header("Introduction")
st.subheader("Gentrification")
st.markdown(
    "Gentrification is a process of urban development in which a neighborhood or portion of a city develops rapidly "
    "in a short period of time, sometimes as a result of urban-renewal programs."
)
st.markdown(
    "A common by-product of gentrification is a deep shift in a neighborhood's socio-residential dynamic, "
    "often characterised by phenomena like the emergence of more expensive housing and new business, "
    "to only cite a couple."
)

st.subheader("The 311 call system")
st.markdown(
    "311 is New York's non-emergency call system and  its mission is to \"provide the public with quick, easy access "
    "to all New York City government services and information while maintaining the highest possible level of "
    'customer service."'
)
st.markdown(
    "Since its launch, NYC311 has consistently been the largest and busiest 311 in North America with an "
    "annual call volume of approximately 21 million, giving us an unparalleled way of gauging the "
    "socio-residential dynamic of specific areas of the city. With this in mind, we pose the following "
    "question in analysing the given datasets:"
)
st.markdown("> Can 311 call data be used to predict gentrification in New York City neighbourhoods?")
st.header("Executive Summary")
st.markdown("In order to build a model, we computed the yearly changes for each type of complaints in every tract. "
            "Furthermore, we considered the ratio of these changes to changes in the city as a whole, as there was a "
            "stable growth in total volume in 331 calls. Finally we considered the ratios of total changes in volumes "
            "of calls in each tract for each type over the given 8 year period. As the types of complaints as well as "
            "the volume are expected to increase due to gentrification, we were expecting to use these parameters to "
            "predict whether a tract was gentrified. ")
st.markdown("Unfortunately, the results did not support our initial "
            "expectations. The below chart shows the correlations between the created variables and whether the tract "
            "gentrified. The variable with the highest correlation is total complaints per capita with correlation of "
            "0.12, which is very low to be conclusive on its own. Further applications of Machine Learning techniques "
            "such as Random Forest, k Nearest Neighbours, SVC and Linear SVC also produced no positive results.")

st.header("Technical report")
st.subheader("Initial Data Exploration")


def get_gazeteer():
    gazeteer = pd.read_csv(
        "data/gazeteer.txt",
        sep="\t",
    )
    lookup_table = {}
    gazeteer.apply(lambda x: lookup_table.__setitem__(x.GEOID, (x.INTPTLAT, x.INTPTLONG)), axis=1)
    return lookup_table


table = get_gazeteer()
adjustments = [1.00, 1.03, 1.05, 1.07, 1.09, 1.09,
               1.10, 1.12, 1.15]


def load_311_data():
    data_311 = [pd.read_csv("data/311Calls/with_tract/2010.csv", index_col=0)]
    data_311[0]["ComplaintYear"] = 2010
    for yr in range(2011, 2019):
        new_data = pd.read_csv(f"data/311Calls/with_tract/{yr}.csv", index_col=0)
        new_data["ComplaintYear"] = yr
        new_data["Longitude"].dropna(inplace=True)
        new_data["Latitude"].dropna(inplace=True)
        data_311.append(new_data)
    return data_311


def fetch_census_data():
    all_census = [pd.read_csv(f"data/census/{x}.csv") for x in range(2010, 2019)]
    for i, census in enumerate(all_census):
        all_census[i] = all_census[i][
            all_census[i]["county"].isin(["081", "085", "005", "061", "047"])
        ]
        all_census[i] = all_census[i][
            all_census[i]["geoid"].isin(table.keys())
        ]
        all_census[i] = all_census[i].set_index(all_census[i].geoid)
        all_census[i].sort_index(inplace=True)
        remove_home_value = all_census[i].index[all_census[i].medianHomeValue == -666666666]
        remove_house_income = all_census[i].index[all_census[i].medianHouseIncome == -666666666]
        for j in range(len(all_census)):
            all_census[j].drop(remove_home_value, inplace=True, errors='ignore')
            all_census[j].drop(remove_house_income, inplace=True, errors='ignore')
        all_census[i]["adjustedHomeValue"] = all_census[i]["medianHomeValue"] / adjustments[i]

        all_census[i]["isEligible"] = compute_eligibility(all_census, i)
        compute_geodata(all_census, i)

        all_census[i]["fractionEducated"] = compute_education(all_census, i)
    common_indices = reduce(np.intersect1d, [census.index for census in all_census])
    for i in range(len(all_census)):
        all_census[i] = all_census[i].loc[common_indices]
    all_census[-1]["hasGentrified"] = compute_gentrification(all_census, -1, 8)
    return all_census


def get_census_data(yr):
    return census_data[yr - 2010]


def compute_education(all_census, i):
    bachelors = all_census[i][f'maleBachelors'] + all_census[i][f'femaleBachelors']
    masters = all_census[i][f'maleMasters'] + all_census[i][f'femaleMasters']
    doctorates = all_census[i][f'maleDoctoral'] + all_census[i][f'femaleDoctoral']
    return (bachelors + masters + doctorates) / (
            all_census[i][f'maleOver25'] + all_census[i][f'femaleOver25'])


def compute_gentrification(all_census, i, years_apart):
    educationPercentChange = 100 * (all_census[i].fractionEducated - all_census[i - years_apart].fractionEducated) / \
                             all_census[i - years_apart].fractionEducated
    educationPercentChangeQuantile = (educationPercentChange >= educationPercentChange.quantile(1 / 2))
    increasedHomeValue = (
            all_census[i][f'adjustedHomeValue'] >= all_census[i - years_apart][f'adjustedHomeValue'])
    homeValuePercentChange = 100 * (all_census[i][f'adjustedHomeValue'] - all_census[i - years_apart][
        f'adjustedHomeValue']) / all_census[i - years_apart][f'adjustedHomeValue']
    homePercentChangeQuantile = (homeValuePercentChange >= homeValuePercentChange.quantile(1 / 2))
    return True & increasedHomeValue & homePercentChangeQuantile & (
            homeValuePercentChange > 0)


def compute_geodata(all_census, i):
    all_census[i]["Latitude"] = all_census[i].apply(lambda tract: table[tract.geoid][0], axis=1)
    all_census[i]["Longitude"] = all_census[i].apply(lambda tract: table[tract.geoid][1], axis=1)
    all_census[i]["h3"] = all_census[i].apply(lambda tract: h3.geo_to_h3(tract.Latitude, tract.Longitude, 9),
                                              axis=1)


def compute_eligibility(all_census, i):
    population_criterion = all_census[i].population >= 500
    income_criterion = all_census[i].medianHouseIncome <= all_census[
        i
    ].medianHouseIncome.quantile(0.5)
    home_value_criterion = all_census[i].medianHomeValue <= all_census[
        i
    ].medianHomeValue.quantile(0.5)
    return population_criterion & income_criterion & home_value_criterion


census_data = fetch_census_data()

st.subheader("Raw Data")
if st.checkbox("Show raw data"):
    year = st.select_slider("Select a year", options=range(2010, 2019), key="a")
    data = get_census_data(year)
    st.write(data)

st.subheader("Defining and quantifying gentrification")
st.markdown(
    "To be able to apply data analysis techniques, we first need to decide on a formal method to decide if a "
    "given tract has been gentrified over time."
)
st.markdown(
    "We shall use a methodology similar to the one outlined by Columbia University professor Lance Freeman in a 2005 "
    "paper (one of the most cited studies on the subject of gentrification). The method is as follows:"
)

col1, col2 = st.beta_columns(2)

col1.markdown("#### Test 1: Is a tract eligible to gentrify?")
col1.markdown(
    "- The tract had a population of at least 500 residents at the beginning and end of a decade. \n "
    "- The tract's median household income was in the bottom 40% compared to all tracts within its "
    "metro area. \n"
    "- The tract's median home value was in the bottom 40% compared to all tracts within its metro "
    "area."
)

col2.markdown("#### Test 2: Has a tract actually gentrified?")
col2.markdown(
    "- An increase in the tract's educational attainment, was in the top 33% of all tracts within the metro area. \n "
    "- The tract’s median home value increased when adjusted for inflation \n"
    "- The percentage increase in the inflation-adjusted median home value was in the top 33% of "
    "all tracts within the metro area."
)

with st.beta_expander("What is 'Educational Attainment'?"):
    st.markdown(
        "For the purpose of this study, we measure educational attainment as the percentage of adults over the age of "
        "25 that hold a Bachelor's degree or higher."
    )

st.subheader("Computing tract gentrification")

col3, col4 = st.beta_columns([1, 2])
col3.markdown("#### Test 1: Eligibility")
ada = col3.select_slider("Select a year", options=range(2010, 2019), key="b")

eligibility_labels = ["Not Eligible", "Eligible"]
eligibility_values = get_census_data(ada).isEligible.value_counts()
eligibility_colors = ["red", "blue"]

fig1 = go.Figure(
    data=[go.Pie(labels=eligibility_labels, values=eligibility_values, marker=dict(colors=eligibility_colors))])

layer1 = pdk.Layer(
    "H3HexagonLayer",
    get_census_data(ada)[["isEligible", "NAME", "h3"]],
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="h3",
    get_fill_color="[255 - 255 * isEligible, 0, 255 * isEligible]",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=0,
)

view_state1 = pdk.ViewState(
    latitude=40.68, longitude=-73.98, zoom=10, bearing=0, pitch=30
)
deck1 = pdk.Deck(initial_view_state=view_state1, layers=[layer1], tooltip={"text": "Tract: {NAME}"})
col3.plotly_chart(fig1, use_container_width=True)
col4.pydeck_chart(deck1)
col3.markdown("#### Test 2: Actual Gentrification (computed for 2018 only to conform to the decade criterion)")

gentrification_labels = ["Not Gentrified", "Gentrified"]
gentrification_colors = ["blue", "#00ff00"]
gentrification_values = get_census_data(2018)[get_census_data(2018).isEligible].hasGentrified.value_counts()

fig2 = go.Figure(data=[
    go.Pie(labels=gentrification_labels, values=gentrification_values, marker=dict(colors=gentrification_colors))])
col3.plotly_chart(fig2, use_container_width=True)
layer2 = pdk.Layer(
    "H3HexagonLayer",
    get_census_data(2018)[get_census_data(2018).isEligible][["hasGentrified", "NAME", "h3"]],
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="h3",
    get_fill_color="[0, 255 * hasGentrified, 255 - 255 * hasGentrified]",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=0,
)

view_state2 = pdk.ViewState(
    latitude=40.68, longitude=-73.98, zoom=10, bearing=0, pitch=30
)
deck2 = pdk.Deck(initial_view_state=view_state2, layers=[layer2], tooltip={"text": "Tract: {NAME}"})
col4.pydeck_chart(deck2)

data_311 = load_311_data()
big_df = [data_311[yr - 2010][data_311[yr - 2010]['geoid'] != -1].merge(get_census_data(yr).reset_index(drop=True),
                                                                        on='geoid') for yr in range(2010, 2019)]

st.subheader("311 calls")
st.markdown(
    "The 311 dataset is very vast and harbors a very diverse set of complaints (180 unique types of complaints). To "
    "enable an efficient analysis, we tried to focus only on the 10 most popular types of complaints for each year. "
)

st.markdown(
    "Our main goal was to try and find a correlation between the nature of the 311 calls and the socio-economic "
    "characteristics of a tract. To do that, we considered variables such as the median home value within a tract and "
    "the median household annual income. "
)

st.markdown(
    "Please note that the following visualisation only consider a 20k random sample of all 311 calls."
)


def get_chart_data(yr):
    return pd.DataFrame(big_df[yr - 2010]["Complaint Type"].value_counts().head(10))


def get_popular_complaints(yr):
    return big_df[yr - 2010]['Complaint Type'].value_counts().head(10).index


def get_mean_income_complaints(yr):
    complaint_mean_income_dict = {}
    for complaint in get_popular_complaints(yr):
        complaint_mean_income_dict[complaint] = \
            big_df[yr - 2010].loc[big_df[yr - 2010]['Complaint Type'] == complaint][
                "medianHouseIncome"].median()

    return pd.Series({k: v for k, v in sorted(complaint_mean_income_dict.items(), key=lambda item: item[1])})


def get_mean_home_value_complaints(yr):
    complaint_median_home_value_dict = {}
    for complaint in get_popular_complaints(yr):
        complaint_median_home_value_dict[complaint] = \
            big_df[yr - 2010].loc[big_df[yr - 2010]['Complaint Type'] == complaint][
                "medianHomeValue"].median()

    return pd.Series({k: v for k, v in sorted(complaint_median_home_value_dict.items(), key=lambda item: item[1])})


afa = st.select_slider("Select a year", options=range(2010, 2019), key="c")

layer = pdk.Layer(
    "GridLayer", big_df[afa - 2010][["Complaint Type", "Latitude_y", "Longitude_y"]].dropna(), pickable=True,
    extruded=True, cell_size=200,
    elevation_scale=4,
    get_position=["Longitude_y", "Latitude_y"],
)

view_state3 = pdk.ViewState(
    latitude=40.68, longitude=-73.98, zoom=10, bearing=0, pitch=45
)
deck3 = pdk.Deck(layers=[layer], initial_view_state=view_state3, tooltip={"text": "{count} complaints"}, )

st.pydeck_chart(deck3)

col5, col6, col7 = st.beta_columns(3)

col5.markdown("#### Annual aggregated count of 311 calls vs. complaint type")
fig2 = px.bar(get_chart_data(afa))
fig2.update_layout(xaxis={'categoryorder': 'total descending'}, showlegend=False, xaxis_title=None,
                   yaxis_title=None)
fig2.update_xaxes(
    tickangle=45)
col5.plotly_chart(fig2, height=1000, use_container_width=True)

col6.markdown("#### Median household income vs. complaint type")
fig3 = px.bar(get_mean_income_complaints(afa))
fig3.update_layout(xaxis={'categoryorder': 'total descending'}, showlegend=False, xaxis_title=None,
                   yaxis_title=None)
fig3.update_xaxes(
    tickangle=45)
col6.plotly_chart(fig3, height=1000, use_container_width=True)

col7.markdown("#### Median home value vs. complaint type")
fig4 = px.bar(get_mean_home_value_complaints(afa))
fig4.update_layout(xaxis={'categoryorder': 'total descending'}, showlegend=False, xaxis_title=None,
                   yaxis_title=None)
fig4.update_xaxes(
    tickangle=45)
col7.plotly_chart(fig4, height=1000, use_container_width=True)

st.markdown(
    "The plots seem to suggest a correlation between a certain type of complaints and a higher household income/home "
    "value; both of which are, by definition, a strong indicator of gentrification. This observation strengthens our "
    "hypothesis that 311 calls could be used to predict gentrification.")

st.markdown("We now need to figure out whether the call "
            "data is indeed a leading indicator or if it's just a trailing indicator of gentrification.")

st.subheader("Exploring the evolution of complaints in areas undergoing gentrification")
st.markdown(
    "On the left-hand side, we we analysed how the gentrification of areas affected their volume of the 311 calls. As "
    "can be seen on the regression on the left, as the tracts underwent gentrification, their volume of 311 calls per "
    "capita increased at a faster rate than in city in general.")
st.markdown(
    "On the right-hand side, however, by plotting the Spearman's rank correlation coefficient of gentrification vs. "
    "complaint type (middle figure) and gentrification vs. per-capita normalised change in complaint type (rightmost figure), "
    "we reach a clear conclusion that the evolution of 311 calls is extremely weakly correlated to gentrification.")

st.markdown(
    "Therefore, we reach the conclusion that 311 calls cannot be an accurate leading indicator for gentrification.")
from PIL import Image

image1 = Image.open('static/images/1.png')
image2 = Image.open('static/images/2.png')
image3 = Image.open('static/images/3.png')
col10, col11, col12 = st.beta_columns(3)
col10.image(image1, use_column_width=True)
col11.image(image2, use_column_width=True)
col12.image(image3, use_column_width=True)
