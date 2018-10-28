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
        if self.input_file:
            input_dataset = gdal.Open(self.input_file, gdal.GA_ReadOnly)
        else:
            raise Exception("No input file was specified")

        log.info("Input file: ( %sP x %sL - %s bands)" % (input_dataset.RasterXSize,
                                                           input_dataset.RasterYSize,
                                                           input_dataset.RasterCount))

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
        out_drv = None
        mem_drv = None
        out_srs = None
        tsize = None
        alphaband = None
        dataBandsCount = None
        out_gt = None
        ominx = None
        omaxx = None
        omaxy = None
        ominy = None

        # How big should be query window be for scaling down
        # Later on reset according the chosen resampling algorightm
        querysize = (query.size[0]*4, query.size[1]*4)

        # Tile format
        tiledriver = 'PNG'
        tileext = 'png'

        # Should we read bigger window of the input raster and scale it down?
        # Note: Modified later by open_input()
        # Not for 'near' resampling
        # Not for Wavelet based drivers (JPEG2000, ECW, MrSID)
        # Not for 'raster' profile
        scaledquery = True

        if self.resampling == 'near':
            querysize = (query.size[0], query.size[1])

        elif self.resampling == 'bilinear':
            querysize = (query.size[0]*2, query.size[1]*2)
        out_drv = gdal.GetDriverByName(tiledriver)
        mem_drv = gdal.GetDriverByName('MEM')

        if not out_drv:
            raise Exception("The '%s' driver was not found, is it available in this GDAL build?",
                            tiledriver)
        if not mem_drv:
            raise Exception("The 'MEM' driver was not found, is it available in this GDAL build?")

        # Open the input file

        input_dataset = gdal.Open(self.input_file, gdal.GA_ReadOnly)

        in_nodata = setup_no_data_values(input_dataset)

        log.info("Preprocessed file: ( %sP x %sL - %s bands)" % (input_dataset.RasterXSize,
                                                                  input_dataset.RasterYSize,
                                                                  input_dataset.RasterCount))

        # TODO out_gt should be based on the SRS of the request
        # Assume WGS84 for now
        out_srs = setup_output_srs(self.in_srs, query)

        ds = input_dataset
        tilebands = self.dataBandsCount + 1

        log.info("dataBandsCount: %s", self.dataBandsCount)
        log.info("tilebands: %s", tilebands)

        (rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize) = geo_query(ds, query.bbox[0], query.bbox[3], query.bbox[2], query.bbox[1], query.size)

        tile_detail = TileDetail(
            rx=rx, ry=ry, rxsize=rxsize, rysize=rysize, wx=wx,
            wy=wy, wxsize=wxsize, wysize=wysize, querysize=querysize,
        )

        conf = TileJobInfo(
            src_file=self.input_file,
            nb_data_bands=self.dataBandsCount,
            tile_extension='png',
            tile_driver='PNG',
            tile_size=query.size,
            in_srs_wkt=self.in_srs_wkt,
            out_geo_trans=self.out_gt,
        )

        return render_request(conf, tile_detail, self.resampling, query.size)


def render_request(tile_job_info, tile_detail, resampling, size):
    gdal.AllRegister()

    dataBandsCount = tile_job_info.nb_data_bands
    tileext = tile_job_info.tile_extension
    tilesize = tile_job_info.tile_size
    options = tile_job_info.options

    tilebands = dataBandsCount + 1
    ds = gdal.Open(tile_job_info.src_file, gdal.GA_ReadOnly)
    mem_drv = gdal.GetDriverByName('MEM')
    out_drv = gdal.GetDriverByName(tile_job_info.tile_driver)
    alphaband = ds.GetRasterBand(1).GetMaskBand()

    rx = tile_detail.rx
    ry = tile_detail.ry
    rxsize = tile_detail.rxsize
    rysize = tile_detail.rysize
    wx = tile_detail.wx
    wy = tile_detail.wy
    wxsize = tile_detail.wxsize
    wysize = tile_detail.wysize
    querysize = tile_detail.querysize

    # Tile dataset in memory
    tilefilename = os.path.join("output.%s" % tileext)
    dstile = mem_drv.Create('', tilesize[0], tilesize[1], tilebands)

    data = alpha = None

    log.info("\tReadRaster Extent: %s, %s" % ((rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize)))

    # Query is in 'nearest neighbour' but can be bigger in then the tilesize
    # We scale down the query to the tilesize by supplied algorithm.

    if rxsize != 0 and rysize != 0 and wxsize != 0 and wysize != 0:
        data = ds.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize,
                             band_list=list(range(1, dataBandsCount + 1)))
        alpha = alphaband.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize)

    # The tile in memory is a transparent file by default. Write pixel values into it if
    # any
    if data:
        if tilesize == querysize:
            # Use the ReadRaster result directly in tiles ('nearest neighbour' query)
            dstile.WriteRaster(wx, wy, wxsize, wysize, data,
                               band_list=list(range(1, dataBandsCount + 1)))
            dstile.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])

            # Note: For source drivers based on WaveLet compression (JPEG2000, ECW,
            # MrSID) the ReadRaster function returns high-quality raster (not ugly
            # nearest neighbour)
            # TODO: Use directly 'near' for WaveLet files
        else:
            # Big ReadRaster query in memory scaled to the tilesize - all but 'near'
            # algo
            dsquery = mem_drv.Create('', querysize[0], querysize[1], tilebands)
            # TODO: fill the null value in case a tile without alpha is produced (now
            # only png tiles are supported)
            dsquery.WriteRaster(wx, wy, wxsize, wysize, data,
                                band_list=list(range(1, dataBandsCount + 1)))
            dsquery.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])

            scale_query_to_tile(dsquery, dstile, tile_job_info.tile_driver, size,
                                tilefilename=tilefilename)
            del dsquery

    # Force freeing the memory to make sure the C++ destructor is called and the memory as well as
    # the file locks are released
    del ds
    del data

    if resampling != 'antialias':
        # Write a copy of tile to png/jpg
        return create_img(dstile, size, tilebands)

    del dstile

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

    raises Gdal2TilesError if the dataset does not contain anything inside this geo_query
    """
    log.info("Geo Query Request: %s %s %s %s %s" % (ulx, uly, lrx, lry, size))
    geotran = ds.GetGeoTransform()
    log.info(geotran)
    rx = int((ulx - geotran[0]) / geotran[1] + 0.001)
    ry = int((uly - geotran[3]) / geotran[5] + 0.001)
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

def scale_query_to_tile(dsquery, dstile, tiledriver, resampling, size, tilefilename=''):
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
                exit_with_error("RegenerateOverview() failed on %s, error %d" % (
                    tilefilename, res))

    elif resampling == 'antialias':

        # Scaling by PIL (Python Imaging Library) - improved Lanczos
        array = numpy.zeros((querysize_x, querysize_y, tilebands), numpy.uint8)
        for i in range(tilebands):
            array[:, :, i] = gdalarray.BandReadAsArray(dsquery.GetRasterBand(i + 1),
                                                       0, 0, querysize_x, querysize_y)
        im = Image.fromarray(array, 'RGBA')     # Always four bands
        im1 = im.resize((tilesize_x, tilesize_y), Image.ANTIALIAS)
        if os.path.exists(tilefilename):
            im0 = Image.open(tilefilename)
            im1 = Image.composite(im1, im0, im1)
        im1.save(tilefilename, tiledriver)

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
        dsquery.SetGeoTransform((0.0, tilesize / float(querysize), 0.0, 0.0, 0.0,
                                 tilesize / float(querysize)))
        dstile.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        res = gdal.ReprojectImage(dsquery, dstile, None, None, gdal_resampling)
        if res != 0:
            exit_with_error("ReprojectImage() failed on %s, error %d" % (tilefilename, res))


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
