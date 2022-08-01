import dclimate_zarr_client.client as client

def test_geo_query(polygon_mask):
    point = client.geo_temporal_query(ipns_name = 'era5_wind_100m_u-hourly', 
        point_kwargs={"lat" : 39.75, "lon" : -118.5},
        rolling_agg_kwargs = {"window_size" : 5, "agg_method" : "mean"})
    rectangle = client.geo_temporal_query(ipns_name = 'era5_wind_100m_u-hourly', 
        rectangle_kwargs={"min_lat" : 39.75, "min_lon" : -120.5, "max_lat" : 40.25, "max_lon" : -119.5}, 
        spatial_agg_kwargs = {"agg_method" : "max"})
    rectangle_nc = client.geo_temporal_query(ipns_name = 'era5_wind_100m_u-hourly', 
        rectangle_kwargs={"min_lat" : 39.75, "min_lon" : -120.5, "max_lat" : 40.25, "max_lon" : -119.5}, 
        spatial_agg_kwargs = {"agg_method" : "max"},
        output_format="netcdf")
    circle = client.geo_temporal_query(ipns_name = 'era5_wind_100m_u-hourly', 
        circle_kwargs={"center_lat" : 40, "center_lon" : -120, "radius" : 150},
        spatial_agg_kwargs = {"agg_method" : "std"},
        temporal_agg_kwargs = {"time_period" : "day", "agg_method" : "std", "time_unit" : 1})
    polygon = client.geo_temporal_query(ipns_name = 'era5_wind_100m_u-hourly', 
        polygon_kwargs={"polygons_mask" : polygon_mask, "epsg_crs" : "epsg:4326"},
        spatial_agg_kwargs = {"agg_method" : "mean"},
        rolling_agg_kwargs = {"window_size" : 5, "agg_method" : "mean"})

    assert point[0] == -2.013934326171875
    assert rectangle[0] == -1.7886962890625
    assert rectangle_nc[0] == 67
    assert circle[0] == 0.44366344809532166
    assert polygon[0] == -1.1927716255187988
