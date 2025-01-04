# Author: Piotr Ostaszewski (325697)
# Created: 2024-12-14T10:09:57.111Z

from osgeo import gdal
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
from osgeo import ogr

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
    raster_norm = (raster_no_inf - raster_no_inf.min()) / (raster_no_inf.max() - raster_no_inf.min())
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

def create_index_raster(index, raster_file, output_name):
    raster_dataset, bands = raster_2_band_arrays(raster_file)
    B = {band + 1 : band for band in range(0, len(bands))}
    if index == "ndvi":
        # oficjalna numeracja kanałów PlanetScope
        R = np.float64(bands[B[6]])
        Nir = np.float64(bands[B[8]])
        # wskaźniczek
        epsilon = 1e-10
        ind = (Nir - R) / (Nir + R + epsilon)
    
    save_with_gdal(ind, raster_dataset, output_name)

    if index == "ndvi":
        ind = ind + 1
    save_with_cv2(ind, output_name)

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
            Flag_of_the_Empty_Array = not Flag_of_the_Empty_Array

        # for each band
        for i, band in enumerate(bands):
            # get stats and weights
            w = w_dict[f"{raster_name}-{i}"]
            s = s_dict[f"{raster_name}-{i}"]
            
            # process the band
            if w["count_type"] == "stats_range":
                new = abs((band - float(s["mean"])) / (float(s["std"]) * float(w["std_range_multiplier"])))
            elif w["count_type"] == "custom_range":
                w_r = w["custom_range"].split(";")
                range_middle = (float(w_r[0]) + float(w_r[1])) / 2
                range_width = abs(float(w_r[0]) - float(w_r[1]))
                new = abs((band - range_middle) / range_width)
            # możemy sobie doimplementować też np. klasyfikację zerojedynkową, czy cokolwiek chcemy
            
            # 0-1 range, values outside range ommitted, multiplied by weight
            processed = processed + (np.where(new < 1, 1-new, 0)) * w["weight"]
    
    # filter by treshold
    processed = np.where(processed > treshold, processed, 0)
    show_grayscale_matplotlib(processed)

    # tif
    save_with_gdal(processed, raster_dataset, output_name)
    
    # png
    save_with_cv2(processed, output_name)

# ***************************************************************************

def read_raster(raster_path):
    src_ds = gdal.Open(raster_path)
    srcband = src_ds.GetRasterBand(1)
    array = srcband.ReadAsArray()
    return src_ds, srcband, array

def normalize_array(array):
    raster_no_nan = np.where(np.isnan(array), 0, array)
    raster_no_inf = np.where(np.isinf(raster_no_nan), 0, raster_no_nan)
    raster_min = raster_no_inf.min()
    raster_max = raster_no_inf.max()
    epsilon = 1e-10
    normalized_array = (raster_no_inf - raster_min) / (raster_max - raster_min + epsilon)
    return normalized_array

def create_binary_array(normalized_array):
    binary_array = np.where(normalized_array > 0, 1, 0)
    return binary_array

def display_binary_array(binary_array):
    plt.imshow(binary_array, cmap='gray')
    plt.title("Binarna wersja rastra")
    plt.colorbar(label="Wartości (0 lub 1)")
    plt.show()

def create_binary_raster(src_ds, binary_array):
    driver = gdal.GetDriverByName("MEM")
    binary_ds = driver.Create("", binary_array.shape[1], binary_array.shape[0], 1, gdal.GDT_Byte)
    binary_ds.SetProjection(src_ds.GetProjection())
    binary_ds.SetGeoTransform(src_ds.GetGeoTransform())
    binary_band = binary_ds.GetRasterBand(1)
    binary_band.WriteArray(binary_array)
    binary_band.FlushCache()
    return binary_ds, binary_band

def create_shapefile(output_shapefile, binary_band):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shapefile):
        drv.DeleteDataSource(output_shapefile)
    dst_ds = drv.CreateDataSource(output_shapefile)
    dst_layer = dst_ds.CreateLayer("polygons", srs=None)
    fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0
    gdal.Polygonize(binary_band, None, dst_layer, dst_field, [], callback=None)
    return dst_ds

def process_shapefile(output_shapefile):
    gdf = gpd.read_file(output_shapefile)
    gdf = gdf[gdf['DN'] != 0]
    dissolved_gdf = gdf.dissolve(by='DN')
    dissolved_gdf.to_file(output_shapefile)

def raster_to_polygon(raster_path, output_shapefile):
    src_ds, srcband, array = read_raster(raster_path)
    normalized_array = normalize_array(array)
    binary_array = create_binary_array(normalized_array)
    display_binary_array(binary_array)
    binary_ds, binary_band = create_binary_raster(src_ds, binary_array)
    dst_ds = create_shapefile(output_shapefile, binary_band)
    
    # Close datasets
    dst_ds = None
    src_ds = None
    binary_ds = None
    
    process_shapefile(output_shapefile)

def split_polygons_to_features(input_shapefile: str, output_shapefile: str):
    gdf = gpd.read_file(input_shapefile)
    exploded_gdf = gdf.explode(index_parts=False)
    exploded_gdf.reset_index(drop=True, inplace=True)
    exploded_gdf.to_file(output_shapefile)


if __name__ == "__main__":
    objects = "data/searchers/searchers.shp"
    objects_old = "data/drogi_prawe/drogi_prawe.shp"

    rasters = {
        "PlanetScope": "data/grupa_1.tif",
        "ndvi": "output/ndvi.tif"
    }

    rasters_2 = {
        "PlanetScope": "data/grupa_2.tif",
        "ndvi": "output/ndvi_2.tif"
    }

    create_index_raster("ndvi", rasters["PlanetScope"], "output/ndvi")
    calculate_raster_stats(rasters, objects_old, generate_weights_json=False)
    detect(rasters, 2)
    raster_to_polygon("output/detected.png", "output/drogi.shp")
    split_polygons_to_features("output/drogi.shp", "output/drogi_rozbite.shp")
