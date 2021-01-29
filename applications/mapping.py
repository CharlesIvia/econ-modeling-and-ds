import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import geometry
from shapely.geometry import Point
import qeds

# activate plot theme
qeds.themes.mpl_style()

# South American cities and their coordinates

df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
    }
)

# Goal: Turn above data into sth plottable - GeoDataFrame

# To mpa the cities we need tuples of coordinates

df["Coordinates"] = list(zip(df.Longitude, df.Latitude))
print(df.head())

# Next: Turn tuple into a Shapely Point object
# Do this by applying Shapely's Point method to the Coordinates column

df["Coordinates"] = df["Coordinates"].apply(Point)
print(df.head())

gdf = gpd.GeoDataFrame(df, geometry="Coordinates")
print(gdf.head())

# Make sure it is a GeoDataFrame

print("ddf is of type:", type(gdf))

# Determine which is the geometry column

print("The geometry column is:", gdf.geometry.name)

# Plotting a map

# This is done in 3 steps: Get the map, Plot the map and lastly Plot the points on the map

# Grab a low resolution world file

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world.set_index("iso_a3")

print(world.head())

# Plotting  SA
fig, gax = plt.subplots(figsize=(10, 10))

# By only plotting rows in which the continent is 'South America' we only plot SA.
world.query("continent == 'South America'").plot(
    ax=gax, edgecolor="black", color="white"
)

# By the way, if you haven't read the book 'longitude' by Dava Sobel, you should...

# Plotting the cities

gdf.plot(ax=gax, color="red", alpha=0.5)

gax.set_xlabel("longitude")
gax.set_ylabel("latitude")
gax.set_title("South America")

gax.spines["top"].set_visible(False)
gax.spines["right"].set_visible(False)

# Label the cities
for x, y, label in zip(gdf["Coordinates"].x, gdf["Coordinates"].y, gdf["City"]):
    gax.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points")


# Case Study: Voting in Winsconsin 2016 Presidential Election

# Find and plot state border

state_df = gpd.read_file(
    "http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_state_5m.zip"
)
print(state_df.head())

print(state_df.columns)

fig, gax = plt.subplots(figsize=(10, 10))
state_df.query("NAME == 'Wisconsin'").plot(ax=gax, edgecolor="black", color="white")

# Finding and plotting county borders

county_df = gpd.read_file(
    "http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_county_5m.zip"
)
print(county_df.head())

print(county_df.columns)

# Wisconsin’s FIPS code is 55 so we will make sure that we only keep those counties.

county_df = county_df.query("STATEFP == '55'")
county_df.plot(ax=gax, edgecolor="black", color="white")

# Get winsonsin vote data

results = pd.read_csv(
    "https://datascience.quantecon.org/assets/data/ruhl_cleaned_results.csv",
    thousands=",",
)
print(results.head())

# Clean the data
results["county"] = results["county"].str.title()
results["county"] = results["county"].str.strip()
county_df["NAME"] = county_df["NAME"].str.title()
county_df["NAME"] = county_df["NAME"].str.strip()

# Merge election results with county data

res_w_states = county_df.merge(results, left_on="NAME", right_on="county", how="inner")

print(res_w_states.head())

# Trump share
res_w_states["trump_share"] = res_w_states["trump"] / (res_w_states["total"])
res_w_states["rel_trump_share"] = res_w_states["trump"] / (
    res_w_states["trump"] + res_w_states["clinton"]
)
print(res_w_states.head())
plt.show()
