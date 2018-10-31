# This file is part of the MapProxy project.
# Copyright (C) 2010 Omniscale <http://omniscale.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mapproxy.layer import MapExtent, DefaultMapExtent, BlankImage, MapLayer
from mapproxy.util.py import reraise_exception
from mapproxy.srs import SRS, get_epsg_num
from osgeo import gdal
from osgeo import osr
import os
from gdal2tiles import TileDetail, TileJobInfo
from mapproxy.image import ImageSource
from mapproxy.image.opts import ImageOptions
from mapproxy.compat import BytesIO

from PIL import Image
import numpy
import osgeo.gdal_array as gdalarray

import logging
log = logging.getLogger('mapproxy.source.gdal')

"""
Retrieve maps/information from GDAL Sources.
"""
class GdalSource(MapLayer):
    def __init__(self, file, image_opts=None, coverage=None, res_range=None, resampling='near'):
        MapLayer.__init__(self, image_opts=image_opts)
        self.input_file = file
        self.coverage = coverage
        self.res_range = res_range
        if self.coverage:
            self.extent = MapExtent(self.coverage.bbox, self.coverage.srs)
        else:
            self.extent = DefaultMapExtent()
        self.resampling = resampling

        # Do some initial checks to see if we can use the specified file
        gdal.AllRegister()

        input_dataset = gdal.Open(self.input_file, gdal.GA_ReadOnly)
        if not input_dataset:
            # Note: GDAL prints the ERROR message too
            exit_with_error("It is not possible to open the input file '%s'." % self.input_file)

        # Read metadata from the input file
        if input_dataset.RasterCount == 0:
            exit_with_error("Input file '%s' has no raster band" % self.input_file)

        if input_dataset.GetRasterBand(1).GetRasterColorTable():
            exit_with_error(
                "Please convert this file to RGB/RGBA and replace the source file with the result.",
                "From paletted file you can create RGBA file (temp.vrt) by:\n"
                "gdal_translate -of vrt -expand rgba %s temp.vrt\n"
                "then set the file parameter in the source" % self.input_file
            )

        self.in_srs, self.in_srs_wkt = setup_input_srs(input_dataset)
        self.epsg = osr.SpatialReference(wkt=self.in_srs_wkt).GetAttrValue('AUTHORITY',1)

        # Get alpha band (either directly or from NODATA value)
        self.alphaband = input_dataset.GetRasterBand(1).GetMaskBand()
        self.dataBandsCount = nb_data_bands(input_dataset)

        # Read the georeference
        self.out_gt = input_dataset.GetGeoTransform()

        # Test the size of the pixel

        # Report error in case rotation/skew is in geotransform (possible only in 'raster' profile)
        if (self.out_gt[2], self.out_gt[4]) != (0, 0):
            exit_with_error("Georeference of the raster contains rotation or skew. "
                            "Such raster is not supported. Please use gdalwarp first.")

        # Here we expect: pixel is square, no rotation on the raster

        input_dataset = None

    def get_map(self, query):
        log.info(query)
        if self.res_range and not self.res_range.contains(query.bbox, query.size,
                                                          query.srs):
            raise BlankImage()
        if self.coverage and not self.coverage.intersects(query.bbox, query.srs):
            raise BlankImage()

        data = self.render(query)

        return ImageSource(data, size=query.size)

    def render(self, query):
        """Constructor function - initialization"""
        # How big should be query window be for scaling down
        # Later on reset according the chosen resampling algorightm
        querysize = (query.size[0]*4, query.size[1]*4)
        if self.resampling == 'near':
            querysize = (query.size[0], query.size[1])
        elif self.resampling == 'bilinear':
            querysize = (query.size[0]*2, query.size[1]*2)

        bbox = transform_bounding_box(query.bbox, get_epsg_num(query.srs.srs_code), int(self.epsg))
        log.info(bbox)

        # Open the input file
        ds = gdal.Open(self.input_file, gdal.GA_ReadOnly)
        in_nodata = setup_no_data_values(ds)

        # Calculate Query
        (rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize) = geo_query(ds, bbox[0], bbox[3], bbox[2], bbox[1], query.size)
        log.info("ReadRaster Extent: %s, %s" % ((rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize)))

        data = alpha = None
        if rxsize != 0 and rysize != 0 and wxsize != 0 and wysize != 0:
            data = ds.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize,
                                 band_list=list(range(1, self.dataBandsCount + 1)))
            alphaband = ds.GetRasterBand(1).GetMaskBand()
            alpha = alphaband.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize)

        if not data:
            raise BlankImage()

        tilebands = self.dataBandsCount + 1
        mem_drv = gdal.GetDriverByName('MEM')
        if not mem_drv:
            raise Exception("The 'MEM' driver was not found, is it available in this GDAL build?")
        dstile = mem_drv.Create('', query.size[0], query.size[1], tilebands)
        if query.size[0] == querysize[0] and query.size[1] == querysize[1]:
            dstile.WriteRaster(wx, wy, wxsize, wysize, data,
                               band_list=list(range(1, self.dataBandsCount + 1)))
            dstile.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])
            # Note: For source drivers based on WaveLet compression (JPEG2000, ECW,
            # MrSID) the ReadRaster function returns high-quality raster (not ugly
            # nearest neighbour)
            # TODO: Use directly 'near' for WaveLet files
        else:
            # User specified a different resampling, so read the larger query size
            # for use by the resampling algorithm
            dsquery = mem_drv.Create('', querysize[0], querysize[1], tilebands)
            dsquery.WriteRaster(wx, wy, wxsize, wysize, data,
                                band_list=list(range(1, self.dataBandsCount + 1)))
            dsquery.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])
            scale_query_to_tile(dsquery, dstile, self.resampling, query.size)
            # TODO: fill the null value in case a tile without alpha is produced (now
            # only png tiles are supported)

        return create_img(dstile, query.size, tilebands)


def create_img(ds, size, bands):
    array = numpy.zeros((size[1], size[0], bands), numpy.uint8)
    for i in range(bands):
        array[:, :, i] = gdalarray.BandReadAsArray(ds.GetRasterBand(i + 1),
                                                   0, 0, size[0], size[1])
    return Image.fromarray(array, 'RGBA')    # Always four bands

def geo_query(ds, ulx, uly, lrx, lry, size=None):
    """
    For given dataset and query in cartographic coordinates returns parameters for ReadRaster()
    in raster coordinates and x/y shifts (for border tiles). If the array is not given, the
    extent is returned in the native resolution of dataset ds.
    """
    log.info("Geo Query Request: %s %s %s %s %s" % (ulx, uly, lrx, lry, size))
    geotran = ds.GetGeoTransform()
    log.info(geotran)
    rx = int((ulx - geotran[0]) / geotran[1])
    ry = int((uly - geotran[3]) / geotran[5])
    rxsize = int((lrx - ulx) / geotran[1] + 0.5)
    rysize = int((lry - uly) / geotran[5] + 0.5)

    if not size:
        wxsize, wysize = rxsize, rysize
    else:
        wxsize, wysize = size[0], size[1]

    # Coordinates should not go out of the bounds of the raster
    wx = 0
    if rx < 0:
        rxshift = abs(rx)
        wx = int(wxsize * (float(rxshift) / rxsize))
        wxsize = wxsize - wx
        rxsize = rxsize - int(rxsize * (float(rxshift) / rxsize))
        rx = 0
    if rx + rxsize > ds.RasterXSize:
        wxsize = int(wxsize * (float(ds.RasterXSize - rx) / rxsize))
        rxsize = ds.RasterXSize - rx

    wy = 0
    if ry < 0:
        ryshift = abs(ry)
        wy = int(wysize * (float(ryshift) / rysize))
        wysize = wysize - wy
        rysize = rysize - int(rysize * (float(ryshift) / rysize))
        ry = 0
    if ry + rysize > ds.RasterYSize:
        wysize = int(wysize * (float(ds.RasterYSize - ry) / rysize))
        rysize = ds.RasterYSize - ry

    return (rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize)

def exit_with_error(message, details=""):
    log.error("error handling request: %s\n" % message)
    if details:
        log.error("\n\n%s\n" % details)

    raise BlankImage()

def scale_query_to_tile(dsquery, dstile, resampling, size):
    """Scales down query dataset to the tile dataset"""
    querysize_x = dsquery.RasterXSize
    querysize_y = dsquery.RasterYSize
    tilesize_x = dstile.RasterXSize
    tilesize_y = dstile.RasterYSize

    tilebands = dstile.RasterCount

    if resampling == 'average':

        # Function: gdal.RegenerateOverview()
        for i in range(1, tilebands + 1):
            # Black border around NODATA
            res = gdal.RegenerateOverview(dsquery.GetRasterBand(i), dstile.GetRasterBand(i),
                                          'average')
            if res != 0:
                exit_with_error("RegenerateOverview() failed, error %d" % res)

    else:

        if resampling == 'near':
            gdal_resampling = gdal.GRA_NearestNeighbour

        elif resampling == 'bilinear':
            gdal_resampling = gdal.GRA_Bilinear

        elif resampling == 'cubic':
            gdal_resampling = gdal.GRA_Cubic

        elif resampling == 'cubicspline':
            gdal_resampling = gdal.GRA_CubicSpline

        elif resampling == 'lanczos':
            gdal_resampling = gdal.GRA_Lanczos

        # Other algorithms are implemented by gdal.ReprojectImage().
        dsquery.SetGeoTransform((0.0, tilesize_x / float(querysize_x), 0.0, 0.0, 0.0,
                                 tilesize_x / float(tilesize_y)))
        dstile.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        res = gdal.ReprojectImage(dsquery, dstile, None, None, gdal_resampling)
        if res != 0:
            exit_with_error("ReprojectImage() failed, error %d" % res)


def setup_no_data_values(input_dataset):
    """
    Extract the NODATA values from the dataset or use the passed arguments as override if any
    """
    in_nodata = []
    for i in range(1, input_dataset.RasterCount + 1):
        raster_no_data = input_dataset.GetRasterBand(i).GetNoDataValue()
        if raster_no_data is not None:
            in_nodata.append(raster_no_data)

    log.info("NODATA: %s" % in_nodata)

    return in_nodata


def setup_input_srs(input_dataset):
    """
    Determines and returns the Input Spatial Reference System (SRS) as an osr object and as a
    WKT representation

    Uses in priority the one passed in the command line arguments. If None, tries to extract them
    from the input dataset
    """

    input_srs = None
    input_srs_wkt = None

    input_srs_wkt = input_dataset.GetProjection()
    if not input_srs_wkt and input_dataset.GetGCPCount() != 0:
        input_srs_wkt = input_dataset.GetGCPProjection()
    if input_srs_wkt:
        input_srs = osr.SpatialReference()
        input_srs.ImportFromWkt(input_srs_wkt)

    return input_srs, input_srs_wkt


def setup_output_srs(input_srs, query):
    """
    Setup the desired SRS (based on options)
    """
    output_srs = osr.SpatialReference()

    # TODO, based on request SRS

    output_srs.ImportFromEPSG(4326)

    return output_srs

def nb_data_bands(dataset):
    """
    Return the number of data (non-alpha) bands of a gdal dataset
    """
    alphaband = dataset.GetRasterBand(1).GetMaskBand()
    if ((alphaband.GetMaskFlags() & gdal.GMF_ALPHA) or
            dataset.RasterCount == 4 or
            dataset.RasterCount == 2):
        return dataset.RasterCount - 1
    else:
        return dataset.RasterCount

def transform_bounding_box(
        bounding_box, base_epsg, new_epsg, edge_samples=11):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Parameters:
        bounding_box (list): a list of 4 coordinates in `base_epsg` coordinate
            system describing the bound in the order [xmin, ymin, xmax, ymax]
        base_epsg (int): the EPSG code of the input coordinate system
        new_epsg (int): the EPSG code of the desired output coordinate system
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.

    Returns:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        `new_epsg` coordinate system.
    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromEPSG(base_epsg)

    new_ref = osr.SpatialReference()
    new_ref.ImportFromEPSG(new_epsg)

    transformer = osr.CoordinateTransformation(base_ref, new_ref)

    p_0 = numpy.array((bounding_box[0], bounding_box[3]))
    p_1 = numpy.array((bounding_box[0], bounding_box[1]))
    p_2 = numpy.array((bounding_box[2], bounding_box[1]))
    p_3 = numpy.array((bounding_box[2], bounding_box[3]))

    def _transform_point(point):
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # This list comprehension iterates over each edge of the bounding box,
    # divides each edge into `edge_samples` number of points, then reduces
    # that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    transformed_bounding_box = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1 - v)) for v in numpy.linspace(
                    0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]
    return transformed_bounding_box
