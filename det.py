# Author: Piotr Ostaszewski (325697)
# Created: 2024-12-14T10:09:57.111Z

from osgeo import gdal, ogr, osr
from typing import Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import geopandas as gpd
import pandas as pd
import shapely
import cv2
from rasterio.features import rasterize
import scipy
import json
import os
import rasterio
# simple GDAL setting
gdal.UseExceptions()

# basic viever
def show_grayscale_matplotlib(array: np.ndarray):  
    plt.imshow(array, cmap='gray')
    plt.show()

# ***************************************************************************

# read spatial raster
def read_spatial_raster(path: Union[str, Path]) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    assert dataset is not None, "Read spatial raster returned None"
    return dataset

# read raster band
def read_raster_band(dataset: gdal.Dataset, band_number: int) -> gdal.Band:
    assert 0 < band_number <= dataset.RasterCount, f"Band number {band_number} is invalid for raster with {dataset.RasterCount} bands."
    band = dataset.GetRasterBand(band_number)
    assert band is not None, f"Unable to read band {band_number}"
    return band

# read band as array
def read_band_as_array(band: gdal.Band) -> np.ndarray:
    array = band.ReadAsArray()
    array = np.copy(array)  # To make sure we do not get memory errors
    return array

# crs to pixels
def points_to_pixels(points: np.ndarray, geotransform: List[float]) -> np.ndarray:
    c, a, _, f, _, e = geotransform
    columns = (points[:, 0] - c) / a
    rows = (points[:, 1] - f) / e
    pixels = np.vstack([rows, columns])
    pixels = pixels.T
    return pixels

# ***************************************************************************

# read features to geopandas
def read_features_to_geopandas(path: Union[str, Path]) -> gpd.GeoDataFrame:
    features = gpd.read_file(path)
    return features

# reproject geodataframe
def reproject_geodataframe(features: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    return features.to_crs(crs)

# crs to pixels
def convert_to_pixel_system(features: gpd.GeoDataFrame, geotransform: List[float]) -> gpd.GeoDataFrame:
    def transform_function(xy: np.ndarray):
        ij = points_to_pixels(xy, geotransform)
        ji = ij[:, [1, 0]]
        return ji
    
    indices = features.index
    for i in indices:
        geometry = features.loc[i, "geometry"]
        geometry = shapely.transform(geometry, transform_function)  # To make our solution work for every type of geometry
        features.loc[i, "geometry"] = geometry
    return features

# ***************************************************************************

def band_lookup_with_objects(raster_file, features_file, band_number="all"):
    bands, object_polygons = pixel_objects_and_bands(raster_file, features_file)
    
    for i, band in enumerate(bands):
        if band_number == "all" or i == band_number:
            plt.imshow(band, cmap='gray')
            for poly in object_polygons:
                plt.plot(*poly.exterior.xy, color='red', linewidth=0.5)
            plt.title("BAND " + str(i + 1))
            plt.show()

def save_with_gdal(array, raster_dataset, output_name):
    array_32 = array.astype(np.float32)
    arr_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_name + ".tif", array_32.shape[1], array_32.shape[0], 1, arr_type)
    out_ds.SetProjection(raster_dataset.GetProjection())
    out_ds.SetGeoTransform(raster_dataset.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array_32)
    band.FlushCache()
    band.ComputeStatistics(False)

def clip_rasters(model_raster, raster_list, output):
    pass

def save_with_cv2(array, output_name):
    raster_no_nan = np.where(np.isnan(array), 0, array)
    raster_no_inf = np.where(np.isinf(raster_no_nan), 0, raster_no_nan)
    raster_norm = (raster_no_inf - raster_no_inf.min()) / (raster_no_inf.max() - raster_no_inf.min() + 1e-10)
    raster_0_255 = raster_norm * 255
    raster_to_save = np.uint8(raster_0_255)
    cv2.imwrite(output_name + ".png", raster_to_save)

def raster_2_band_arrays(raster_file):
    # raster
    raster_dataset = read_spatial_raster(raster_file)
    raster_count = raster_dataset.RasterCount

    # bands
    bands = []
    for i in range(1, raster_count+1):
        band = read_raster_band(raster_dataset, i)
        array = read_band_as_array(band)
        bands.append(array)

    return raster_dataset, bands

def pixel_objects_and_bands(raster_file, features_file):
    # bands
    raster_dataset, bands = raster_2_band_arrays(raster_file)

    # features
    features = read_features_to_geopandas(features_file)
    features = reproject_geodataframe(features, raster_dataset.GetProjection())
    features = convert_to_pixel_system(features, raster_dataset.GetGeoTransform())

    # feature polygons
    object_polygons = features["geometry"].tolist()

    return bands, object_polygons


def create_index_raster(index, input_raster_path, output_raster):
    raster_dataset, bands = raster_2_band_arrays(input_raster_path)
    B = {band + 1 : bands[band] for band in range(len(bands))}
    coastal_blue = np.float64(B[1])
    blue = np.float64(B[2])
    green_i = np.float64(B[3])
    green = np.float64(B[4])
    yellow = np.float64(B[5])
    red = np.float64(B[6])
    rededge = np.float64(B[7])
    nir = np.float64(B[8])

    # Calculate the requested index
    if index == "ndvi":
        index_data = (nir - red) / (nir + red + 1e-10)
    elif index == "ndwi":
        index_data = (green - nir) / (green + nir + 1e-10)
    elif index == "savi":
        L = 0.5  # Soil adjustment factor
        index_data = ((nir - red) * (1 + L)) / (nir + red + L + 1e-10) * (1 + L)
    elif index == "evi":
        G, C1, C2, L = 2.5, 6, 7.5, 1
        index_data = G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-10)
    elif index == "ndre" and rededge is not None:
        index_data = (nir - rededge) / (nir + rededge + 1e-10)
    elif index == "ratio_nir_red":
        index_data = nir / (red + 1e-10)
    elif index == "difference_nir_red":
        index_data = nir - red
    else:
        raise ValueError(f"Unsupported index: {index}")

    save_with_gdal(index_data, raster_dataset, output_raster)

    if index == "ndvi":
        index_data = index_data + 1
    
    save_with_cv2(index_data, output_raster)
    

def calculate_raster_stats(raster_dict, features_file, output_file="output/statistics.json", generate_weights_json=False, weights_name="weights.json"):
    # the output dictionary
    stats_dict = {}

    # all rasters
    for raster_name, raster_path in raster_dict.items():
        bands, object_polygons = pixel_objects_and_bands(raster_path, features_file)
        # all bands
        for i, band in enumerate(bands):
            band_arr = np.empty(0)
            # all objects     
            for poly in object_polygons:
                data_pixels = rasterize([poly], band.shape)
                data_pixels = np.bool_(data_pixels)    
                pixel_values = band[data_pixels]
                band_arr = np.append(band_arr, pixel_values)
            
            # band stats
            minv, maxv, meanv, medianv, stdv = band_arr.min(), band_arr.max(), band_arr.mean(), np.median(band_arr) ,band_arr.std()
            trim_mean = scipy.stats.trim_mean(band_arr, 0.1)
            
            # save to dictionary
            stats_dict[(f"{raster_name}-{i}")] = {
                "BAND" : i + 1,
                "min": minv,
                "max": maxv,
                "mean": meanv,
                "median": medianv,
                "std": stdv,
                "trim_mean": trim_mean
            }

    # save to json        
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(stats_dict, file, indent=4)
        file.write('\n')
    
    # generate 'default' weights json
    if generate_weights_json:
        weights_dict = {}
        for key in stats_dict.keys():
            weights_dict[key] = {
                "BAND": int(key.split("-")[1]) + 1,
                "weight": 1,
                "count_type" : "stats_range",
                "std_range_multiplier": 0.5,
                "custom_range" : "-0.1;0.1"
            }
        # save to json
        with open(weights_name, "w", encoding="utf-8") as file:
            json.dump(weights_dict, file, indent=4)
            file.write('\n')

def detect(raster_dict, treshold, stats="output/statistics.json", weights="weights.json", output_name="output/detected"):
    # load analysis data
    w_dict = json.load(open(weights))
    s_dict = json.load(open(stats))
    
    Flag_of_the_Empty_Array = True
    # do the analysis for each raster
    for raster_name, raster_path in raster_dict.items():
        raster_dataset, bands = raster_2_band_arrays(raster_path)
        
        # create empty array for processed data
        if Flag_of_the_Empty_Array:
            processed = np.zeros((raster_dataset.RasterYSize, raster_dataset.RasterXSize))
            elim = np.ones((raster_dataset.RasterYSize, raster_dataset.RasterXSize))
            Flag_of_the_Empty_Array = not Flag_of_the_Empty_Array

        # for each band
        for i, band in enumerate(bands):
            # get stats and weights
            w = w_dict[f"{raster_name}-{i}"]
            s = s_dict[f"{raster_name}-{i}"]
            
            # process the band
            if w["count_type"] == "stats_range" or w["count_type"] == "stats_range_eliminate":
                new = abs((band - float(s["mean"])) / (float(s["std"]) * float(w["std_range_multiplier"]) + 1e-10))
            elif w["count_type"] == "custom_range":
                w_r = w["custom_range"].split(";")
                range_middle = (float(w_r[0]) + float(w_r[1])) / 2
                range_width = abs(float(w_r[0]) - float(w_r[1]))
                new = abs((band - range_middle) / range_width + 1e-10)

            # 0-1 range, values outside range ommitted, multiplied by weight
            processed = processed + (np.where(new < 1, 1-new, 0)) * w["weight"]

            if w["count_type"] == "stats_range_eliminate":
                elim = elim * np.where(new < 0.5, 1, 0)
    
    # filter by treshold
    processed = processed * elim
    processed = np.where(processed > treshold, processed, 0)
    show_grayscale_matplotlib(processed)

    processed = np.where(processed > 0, 1, 0)
    show_grayscale_matplotlib(processed)
    # tif
    save_with_gdal(processed, raster_dataset, output_name)
    
    # png
    save_with_cv2(processed, output_name)

# ***************************************************************************

def normalize_raster(raster, output):
    src = read_spatial_raster(raster)
    band = read_raster_band(src, 1)
    raster = read_band_as_array(band)

    # normalizacja do 0-1 
    raster_no_nan = np.where(np.isnan(raster), 0, raster)
    raster_no_inf = np.where(np.isinf(raster_no_nan), 0, raster_no_nan)
    raster_norm = (raster_no_inf - raster_no_inf.min()) / (raster_no_inf.max() - raster_no_inf.min() + 1e-10)
    raster_0_1 = np.where(raster_norm > 0.5, 1, 0)
    show_grayscale_matplotlib(raster_0_1)
    save_with_cv2(raster_0_1, output)
    save_with_gdal(raster_0_1, src, output)

# wykrywamoe krawędzi dróg algorytmem Canny'ego
def detect_and_connect_edges(input, output):
    # Wczytanie obrazu w skali szarości
    img = cv2.imread(input + ".png", 0)
    
    # Wykrywanie krawędzi algorytmem Canny'ego
    edges = cv2.Canny(img, 100, 200)
    
    # Zastosowanie operacji morfologicznych
    kernel = np.ones((6, 6), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.erode(edges_dilated, kernel, iterations=1)
    
    # Znajdowanie konturów
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tworzenie pustego obrazu do rysowania konturów
    connected_edges = np.zeros_like(edges)
    
    # Rysowanie konturów na pustym obrazie
    cv2.drawContours(connected_edges, contours, -1, (255), thickness=cv2.FILLED)
    
    # Wyświetlanie obrazu krawędzi
    show_grayscale_matplotlib(connected_edges)
    
    # Zapisywanie obrazu krawędzi za pomocą OpenCV
    save_with_cv2(connected_edges, output)
    
    # Zapisywanie obrazu krawędzi za pomocą GDAL
    save_with_gdal(connected_edges, read_spatial_raster(input + ".tif"), output)


def create_shapefile(input_raster, output):
    src = read_spatial_raster(input_raster)
    band = read_raster_band(src, 1)
    raster = read_band_as_array(band)

    if os.path.exists(output):
        os.remove(output)

    # tworzymy nowy plik shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(output)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src.GetProjection())
    layer = ds.CreateLayer("drogi", srs)
    fd = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(fd)
    field = 0
    gdal.Polygonize(band, None, layer, field, [], callback=None)
    
    return ds

def change_shapefile(output):
    gdf = gpd.read_file(output)
    gdf = gdf[gdf['id'] != 0]
    dissolved = gdf.dissolve(by='id')
    dissolved.to_file(output)
    gdf = gpd.read_file(output)
    exploded_gdf = gdf.explode(index_parts=False)
    exploded_gdf.reset_index(drop=True, inplace=True)
    exploded_gdf.to_file(output)

def buffor_add_shapefile(input, output, buffor):
    gdf = gpd.read_file(input)
    gdf['geometry'] = gdf.buffer(buffor)
    gdf.to_file(output)

def buffor_sub_shapefile(input, output, buffor):
    gdf = gpd.read_file(input)
    gdf['geometry'] = gdf.buffer(-buffor)
    gdf.to_file(output)

def delete_small_objects(input, output, min_area):
    gdf = gpd.read_file(input)
    gdf = gdf[gdf.area > min_area]
    gdf.to_file(output)
    
def connect_nearest_polygon(input, output):
    # Wczytaj dane wejściowe
    gdf = gpd.read_file(input)
    
    # Sprawdź, czy geometrie są poprawne i napraw je, jeśli to konieczne
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # Dodaj nową kolumnę, która przechowa informacje o połączeniu
    gdf['connected'] = False

    for i, row in gdf.iterrows():
        # Znajdź najbliższy poligon
        other_geometries = gdf.loc[gdf.index != i, 'geometry']
        distances = other_geometries.distance(row['geometry'])
        
        if distances.empty:
            continue
        
        # Znajdź indeks najbliższego poligonu
        nearest_idx = distances.idxmin()

        # Połącz geometrie bieżącego i najbliższego poligonu
        nearest_geometry = gdf.loc[nearest_idx, 'geometry']
        gdf.at[i, 'geometry'] = row['geometry'].union(nearest_geometry)
        gdf.at[nearest_idx, 'geometry'] = row['geometry'].union(nearest_geometry)

        # Oznacz oba poligony jako połączone
        gdf.at[i, 'connected'] = True
        gdf.at[nearest_idx, 'connected'] = True

    # Zapisz dane wyjściowe do pliku
    gdf.to_file(output, driver='ESRI Shapefile')

def objects_area_perimeter_filter(source, input, output):
    gdf = gpd.read_file(source)
    
    # pole i obwód dla wszystkich poligonów na podsawie geometrii
    gdf['area'] = gdf['geometry'].area
    gdf['perimeter'] = gdf['geometry'].length
    gdf['area_per'] = gdf['area'] / gdf['perimeter']

    mean = gdf['area_per'].mean()
    std = gdf['area_per'].std()

    # Filtracja obiektów na podstawie stosunku pola do obwodu
    gdf2 = gpd.read_file(input)
    gdf2['area_per'] = gdf2['geometry'].area / gdf2['geometry'].length
    gdf2 = gdf2[gdf2['area_per'] > mean - std * 1.78]
    gdf2 = gdf2[gdf2['area_per'] < mean + std * 1.78]
    gdf2.to_file(output)

if __name__ == "__main__":
    objects = "data/drogi_prawe/drogi_prawe.shp"

    rasters = {
        "PlanetScope": "data/grupa_1.tif",
        "ndvi": "output/ndvi.tif",
        "ndwi": "output/ndwi.tif",
        "savi": "output/savi.tif",
        # "evi": "output/evi.tif",
        "ndre": "output/ndre.tif",
        "ratio_nir_red": "output/ratio_nir_red.tif", # 0
        "difference_nir_red": "output/difference_nir_red.tif"
    }

    # Create various index rasters
    create_index_raster("ndvi", rasters["PlanetScope"], "output/ndvi")
    create_index_raster("ndwi", rasters["PlanetScope"], "output/ndwi")
    create_index_raster("savi", rasters["PlanetScope"], "output/savi")
    create_index_raster("evi", rasters["PlanetScope"], "output/evi")
    create_index_raster("ndre", rasters["PlanetScope"], "output/ndre")
    create_index_raster("ratio_nir_red", rasters["PlanetScope"], "output/ratio_nir_red")
    create_index_raster("difference_nir_red", rasters["PlanetScope"], "output/difference_nir_red")

    # Calculate raster statistics
    calculate_raster_stats(rasters, objects, generate_weights_json=False)

    detect(rasters, 10)
    detect_and_connect_edges("output/detected", "output/edges")
    create_shapefile("output/edges.tif", "output/edges.shp")
    change_shapefile("output/edges.shp")
    buffor_add_shapefile("output/edges.shp", "output/edges_buffor.shp", 15)  # Poprawiona nazwa pliku
    change_shapefile("output/edges_buffor.shp")
    buffor_sub_shapefile("output/edges_buffor.shp", "output/edges_buffor_sub.shp", 15) # Poprawiona nazwa pliku
    change_shapefile("output/edges_buffor_sub.shp")
    delete_small_objects("output/edges_buffor_sub.shp", "output/edges_buffor_sub_del.shp", 500) # Poprawiona nazwa pliku
    change_shapefile("output/edges_buffor_sub_del.shp")
    connect_nearest_polygon("output/edges_buffor_sub_del.shp", "output/edges_buffor_sub_del_con.shp") # Poprawiona nazwa pliku
    objects_area_perimeter_filter(objects, "output/edges_buffor_sub_del_con.shp", "output/edges_area_perim.shp")