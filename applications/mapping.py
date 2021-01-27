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
