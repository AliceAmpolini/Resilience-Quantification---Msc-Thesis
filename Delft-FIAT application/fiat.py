# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:48 2021

@author: L.K. Leunge
"""


from ast import literal_eval
from hydromt import gis_utils, io
from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog
from pathlib import Path
import click
import datetime
import geopandas as gpd
import inspect
import logging
import numpy as np
import pandas as pd
import shutil
import xarray as xr


__version__ = '0.0.1'


logger = logging.getLogger(__name__)


class FiatModel:
    """A model that calculates the expected hazard impact in terms of the direct
    damage per hazard map and (optionally) the expected annual damage (EAD) over
    set of hazard maps."""

    def __init__(
        self,
        root=None,
        config_fn=None,
        logger=logger,
    ):
        self.config_fn = config_fn
        self.set_root(root)
        self.logger = logger
        self.staticmaps = xr.Dataset()
        self.staticgeoms = dict()        # dictionary of gdp.GeoDataFrame
        self.config = dict()             # nested dictionary
        self.results = dict()            # dictionary of xr.DataArray

    """ MODEL WORKFLOW """

    def run(self):
        """Execute the FIAT workflow."""

        # Parse config.
        opt = parse_config(self.config_fn)

        # Check if config method exist.
        opt = self.check_get_opt(opt)

        # Run setup_config.
        self.run_log_method('setup_config', **opt.pop('setup_config', {}))

        # Run the other methods.
        for method in opt:
            self.run_log_method(method, **opt[method])

        # Run setup_geometry.
        self.run_log_method('setup_geometry', self.get_config('region'))

        # Calculate damage.
        self.calculate_damage()

        # Calculate risk.
        self.calculate_risk()

        # Write results.
        self.write_results()

    """ MODEL METHODS """

    def setup_config(self, **cfdict):
        """Update config with a dictionary."""

        if len(cfdict) > 0:
            self.logger.debug(
                'Setting model config options.'
            )
        for key, value in cfdict.items():
            self.update_config(key, value)

    def setup_hazard(
        self,
        usage,
        map_fn,
        map_type,
        chunks='auto',
        rp=None,
        crs=None,
        nodata=None,
        var=None,
    ):
        """Add a hazard map to the FIAT model.

        Adds model layer:

        * **hazard** map(s): A raster map with the nomenclature <file_name>.

        Parameters
        ----------
        Usage: Boolean
            Indicator that specifies whether the layer should be included in the calculation. The default value is True.
        map_fn: (list of) str, Path
            Absolute or relative (with respect to the configuration file or hazard directory) path to the hazard file.
        map_type: (list of) str
            Description of the hazard type.
        rp: (list of) int, float, optional
            Return period in years, required for a risk calculation.
        crs: (list of) int, str, optional
            Coordinate reference system of the hazard file.
        nodata: (list of) int, float, optional
            Value that is assigned as nodata.
        var: (list of) str, optional
            Hazard variable name in NetCDF input files.
        chunks: (list of) int, optional
            Chunk sizes along each dimension used to load the hazard file into a dask array. The default value is 'auto'.
        """

        # Check if layer should be included in the calculation.
        if not usage:
            return

        # Check the hazard input parameter types.
        map_fn_lst = [map_fn] if isinstance(map_fn, (str, Path)) else map_fn
        map_type_lst = [map_type] if isinstance(map_type, (str, Path)) else map_type
        self.check_param_type(map_fn_lst, name='map_fn', types=(str, Path))
        self.check_param_type(map_type_lst, name='map_type', types=str)
        if chunks != 'auto':
            chunks_lst = [chunks] if isinstance(chunks, (int, dict)) else chunks
            self.check_param_type(chunks_lst, name='chunks', types=(int, dict))
            if not len(chunks_lst) == 1 or not len(chunks_lst) == len(map_fn_lst):
                raise IndexError(
                    'The number of "chunks" parameters should match with the number of '
                    '"map_fn" parameters.'
                )
        if rp is not None:
            rp_lst = [rp] if isinstance(rp, (int, float)) else rp
            self.check_param_type(rp_lst, name='rp', types=(float, int))
            if not len(rp_lst) == len(map_fn_lst):
                raise IndexError(
                    'The number of "rp" parameters should match with the number of '
                    '"map_fn" parameters.'
                )
        if crs is not None:
            crs_lst = [str(crs)] if isinstance(crs, (int, str)) else crs
            self.check_param_type(crs_lst, name='crs', types=(int, str))
        if nodata is not None:
            nodata_lst = [nodata] if isinstance(nodata, (int, float)) else nodata
            self.check_param_type(nodata_lst, name='nodata', types=(int, float))
        if var is not None:
            var_lst = [var] if isinstance(var, str) else var
            self.check_param_type(var_lst, name='var', types=str)

        # Check if the hazard input directory exist.
        self.check_dir_exist(self.get_config('hazard_dp'))

        # Check if the hazard input files exist.
        self.check_file_exist(map_fn_lst, name='map_fn', input_dir='hazard_dp')

        # Read the hazard map(s) and add to config and staticmaps.
        for idx, da_map_fn in enumerate(map_fn_lst):
            da_name = da_map_fn.stem
            da_type = self.get_param(
                map_type_lst, map_fn_lst, 'hazard', da_name, idx, 'map type'
            )
            if da_map_fn.suffix == '.tif':
                da = io.open_raster(
                    da_map_fn, chunks=chunks if chunks == 'auto' else chunks_lst[idx],
                )
            elif da_map_fn.suffix == '.nc':
                if var is None:
                    raise ValueError(
                        'The "var" parameter is required when reading NetCDF data.'
                    )
                da_var = self.get_param(
                    var_lst, map_fn_lst, 'hazard', da_name, idx, 'NetCDF variable'
                )
                da = xr.open_dataset(
                    da_map_fn,
                    engine='netcdf4',
                    chunks=chunks if chunks == 'auto' else chunks_lst[idx],
                )[da_var]

            # Set (if necessary) the coordinate reference system.
            if crs is not None and not da.raster.crs.is_epsg_code:
                da_crs = self.get_param(
                    crs_lst, map_fn_lst, 'hazard', da_name, idx,
                    'coordinate reference system',
                )
                da_crs_str = da_crs if 'EPSG' in da_crs else f'EPSG:{da_crs}'
                da.raster.set_crs(da_crs_str)
            elif crs is None and not da.raster.crs.is_epsg_code:
                raise ValueError(
                    'The hazard map has no coordinate reference system assigned.'
                )

            # Set (if necessary) and mask the nodata value.
            if nodata is not None:
                da_nodata = self.get_param(
                    nodata_lst, map_fn_lst, 'hazard', da_name, idx, 'nodata'
                )
                da.raster.set_nodata(nodata=da_nodata)
                da = da.raster.mask_nodata()
            elif nodata is None and da.raster.nodata is None:
                raise ValueError(
                        'The hazard map has no nodata value assigned.'
                    )

            # Correct (if necessary) the grid orientation from the lower to the upper left corner.
            if da.raster.res[1] > 0:
                da = da.reindex({da.raster.y_dim: list(reversed(da.raster.ycoords))})

            # Check if the obtained hazard map is identical.
            if self.staticmaps and not self.staticmaps.raster.identical_grid(da):
                raise ValueError(
                    'The hazard maps should have identical grids.'
                )

            # Get the return period input parameter.
            da_rp = self.get_param(rp_lst, map_fn_lst, 'hazard', da_name, idx, 'return period') if 'rp_lst' in locals() else None
            if self.get_config('risk_output') and da_rp is None:

                # Get (if possible) the return period from dataset names if the input parameter is None.
                if 'rp' in da_name.lower():
                    fstrip = lambda x: x in '0123456789.'
                    rp_str = ''.join(
                        filter(fstrip, da_name.lower().split('rp')[-1])
                    ).lstrip('0')
                    try:
                        assert isinstance(literal_eval(rp_str) if rp_str else None, (int, float))
                        da_rp = literal_eval(rp_str)
                    except AssertionError:
                        raise ValueError(
                            f'Could not derive the return period for hazard map: {da_name}.'
                        )
                else:
                    raise ValueError(
                        'The hazard map must contain a return period in order to conduct a risk calculation.'
                    )

            # Add the hazard map to config and staticmaps.
            hazard_type = self.get_config('hazard_type', fallback='flooding')
            self.check_uniqueness(
                'hazard',
                da_type,
                da_name,
                {
                    'name': da_name,
                    'map_fn': da_map_fn,
                    'rp': da_rp,
                    'chunks': 'auto' if chunks == 'auto' else chunks_lst[idx],
                },
                file_type='hazard',
                filename=da_name,
            )
            self.update_config(
                'hazard',
                da_type,
                da_name,
                {
                    'name': da_name,
                    'map_fn': da_map_fn,
                    'rp': da_rp,
                    'chunks': 'auto' if chunks == 'auto' else chunks_lst[idx],
                },
            )
            self.set_staticmaps(da, da_name)
            post = f'(rp {da_rp})' if rp is not None and self.get_config('risk_output') else ''
            self.logger.info(
                f'Added {hazard_type} hazard map: {da_name} {post}'
            )

    def setup_exposure(
        self,
        usage,
        map_fn,
        category,
        unit,
        function_fn,
        resamp_alg='nearest',
        chunks='auto',
        comp_alg='average',
        scale_factor=1,
        weight_factor=1,
        subcategory=None,
        crs=None,
        nodata=None,
        var=None,
    ):
        """Add an exposure map to the FIAT model.

        Adds model layer:

        * **exposure** map(s): A raster map with the nomenclature <category_name>_<subcategory_name>.

        Parameters
        ----------
        Usage: Boolean
            Indicator that specifies whether the layer should be included in the calculation. The default value is True.
        map_fn: (list of) str, Path
            Absolute or relative (with respect to the configuration file or exposure directory) path to the exposure file.
        category: (list of) str, int, float
            Name of the exposure category.
        subcategory: (list of) str, int, float, optional
            Name of the exposure subcategory.
        unit: (list of) str
            Unit of the exposure map values.
        crs: (list of) int, str, optional
            Coordinate reference system of the exposure file.
        nodata: (list of) float, int, optional
            Value that is assigned as nodata.
        var: (list of) str, optional
            Hazard variable name in NetCDF input files.
        resamp_alg: (list of) str, optional
            Name of the rampling method used to reproject the exposure file into the hazard projection. The default value is 'nearest'.
        chunks: (list of) int, optional
            Chunk sizes along each dimension used to load the exposure file into a dask array. The default value is 'auto'.
        function_fn: (list of) dict
            Absolute or relative (with respect to the configuration file or the susceptibility directory) path to the susceptibility file.
        comp_alg: (list of) str, optional
            Name of the method used to determine the compound damage factor. The default value is 'average'.
        scale_factor: (list of) int, float, optional
            Scaling factor of the exposure values. The default value is 1.
        weight_factor: (list of) int, float, optional
            Weight factor of the exposure values in the total damage and risk results. The default value is 1.
        """

        # Check if layer should be included in the calculation.
        if not usage:
            return

        # Check the exposure input parameter types.
        map_fn_lst = [map_fn] if isinstance(map_fn, (str, Path)) else map_fn
        cat_lst = [category] if isinstance(category, (str, int, float)) else category
        unit_lst = [unit] if isinstance(unit, str) else unit
        fun_fn_lst = [function_fn] if isinstance(function_fn, dict) else function_fn
        self.check_param_type(map_fn_lst, name='map_fn', types=(str, Path))
        self.check_param_type(cat_lst, name='category', types=(str, int, float))
        self.check_param_type(unit_lst, name='unit', types=str)
        self.check_param_type(fun_fn_lst, name='function_fn', types=dict)
        if resamp_alg != 'nearest':
            ra_lst = [resamp_alg] if isinstance(resamp_alg, str) else resamp_alg
            self.check_param_type(ra_lst, name='resamp_alg', types=str)
        if chunks != 'auto':
            chunks_lst = [chunks] if isinstance(chunks, (int, dict)) else chunks
            self.check_param_type(chunks_lst, name='chunks', types=(int, dict))
            if not len(chunks_lst) == 1 or not len(chunks_lst) == len(map_fn_lst):
                raise IndexError(
                    'The number of "chunks" parameters should match with the number of "map_fn" parameters.'
                )
        if comp_alg != 'average':
            comp_alg_lst = [comp_alg] if isinstance(comp_alg, str) else comp_alg
            self.check_param_type(comp_alg_lst, name='comp_alg', types=str)
        if scale_factor != 1:
            sf_lst = [scale_factor] if isinstance(scale_factor, (int, float)) else scale_factor
            self.check_param_type(sf_lst, name='scale_factor', types=(int, float))
        if weight_factor != 1:
            wf_lst = [weight_factor] if isinstance(weight_factor, (int, float)) else weight_factor
            self.check_param_type(wf_lst, name='weight_factor', types=(int, float))
        if subcategory is not None:
            subcat_lst = [subcategory] if isinstance(subcategory, str) else subcategory
            self.check_param_type(subcat_lst, name='subcategory', types=str)
        if crs is not None:
            crs_lst = [str(crs)] if isinstance(crs, (int, str)) else crs
            self.check_param_type(crs_lst, name='crs', types=(int, str))
        if nodata is not None:
            nodata_lst = [nodata] if isinstance(nodata, (float, int)) else nodata
            self.check_param_type(nodata_lst, name='nodata', types=(float, int))
        if var is not None:
            var_lst = [var] if isinstance(var, str) else var
            self.check_param_type(var_lst, name='var', types=str)

        # Check if the exposure and vulnerability input directories exist.
        self.check_dir_exist(self.get_config('exposure_dp'))
        self.check_dir_exist(self.get_config('susceptibility_dp'))

        # Check if the exposure and vulnerability input files exist.
        self.check_file_exist(map_fn_lst, name='map_fn', input_dir='exposure_dp')
        self.check_file_exist(fun_fn_lst, name='function_fn', input_dir='susceptibility_dp')

        # Read the exposure map(s) and add to config and staticmaps.
        for idx, da_map_fn in enumerate(map_fn_lst):
            da_name = da_map_fn.stem
            if da_map_fn.suffix == '.tif':
                da = io.open_raster(
                    da_map_fn,
                    chunks=chunks if chunks == 'auto' else chunks_lst[idx],
                )
            elif da_map_fn.suffix == '.nc':
                if var is None:
                    raise ValueError(
                        'The "var" parameter is required when reading NetCDF data.'
                    )
                da_var = self.get_param(
                    var_lst, map_fn_lst, 'hazard', da_name, idx, 'NetCDF variable'
                )
                da = xr.open_dataset(
                    da_map_fn,
                    engine='netcdf4',
                    chunks=chunks if chunks == 'auto' else chunks_lst[idx],
                )[da_var]

            # Set (if necessary) the coordinate reference system.
            if crs is not None and not da.raster.crs.is_epsg_code:
                da_crs = self.get_param(
                    crs_lst, map_fn_lst, 'exposure', da_name, idx, 'coordinate reference system'
                )
                da_crs_str = da_crs if 'EPSG' in da_crs else f'EPSG:{da_crs}'
                da.raster.set_crs(da_crs_str)
            elif crs is None and not da.raster.crs.is_epsg_code:
                raise ValueError(
                    'The exposure map has no coordinate reference system assigned.'
                )

            # Set (if necessary) and mask the nodata value.
            if nodata is not None:
                da_nodata = self.get_param(
                    nodata_lst, map_fn_lst, 'hazard', da_name, idx, 'nodata'
                )
                da.raster.set_nodata(nodata=da_nodata)
                da = da.raster.mask_nodata()
            elif nodata is None and da.raster.nodata is None:
                raise ValueError(
                        'The exposure map has no nodata value assigned.'
                    )

            # Correct (if necessary) the grid orientation from the lower to the upper left corner.
            if da.raster.res[1] > 0:
                da = da.reindex({da.raster.y_dim: list(reversed(da.raster.ycoords))})

            # Reproject and clip (if necessary) the exposure map to the hazard extent.
            if self.staticmaps and not self.staticmaps.raster.identical_grid(da):
                da_resamp_alg = self.get_param(
                    ra_lst, map_fn_lst, 'exposure', da_name, idx, 'resample algorithm'
                ) if resamp_alg != 'nearest' else resamp_alg
                da = da.raster.clip_bbox(
                    bbox=self.staticmaps.raster.transform_bounds(da.raster.crs),
                    buffer=4,
                )
                da = da.where(
                    da == da,
                    other=0,
                )
                if da_resamp_alg == 'sum':
                    da = self.get_density_grid(da).raster.reproject_like(
                        self.staticmaps, method='average',
                    ) * self.get_area_grid(self.staticmaps)
                else:
                    da = da.raster.reproject_like(
                        self.staticmaps, method=da_resamp_alg,
                    )
                da = da.raster.mask_nodata()

            # Get the category input parameter.
            da_cat = self.get_param(
                cat_lst, map_fn_lst, 'exposure', da_name, idx, 'category'
            )

            # Get the subcategory input parameter.
            da_subcat = self.get_param(
                subcat_lst, map_fn_lst, 'exposure', da_name, idx, 'subcategory'
            ) if subcategory is not None else subcategory

            # Get the unit input parameter.
            da_unit = self.get_param(
                unit_lst, map_fn_lst, 'exposure', da_name, idx, 'unit'
            )

            # Get the function_fn input parameter.
            da_function_fn = self.get_param(
                fun_fn_lst, map_fn_lst, 'exposure', da_name, idx,
                'susceptibility function',
            )

            # Get the comp_alg input parameter.
            da_comp_alg = self.get_param(
                comp_alg_lst, map_fn_lst, 'exposure', da_name, idx,
                'compound algorithm',
            ) if comp_alg != 'average' else comp_alg

            # Get the scale_factor input parameter.
            da_sf = self.get_param(
                sf_lst, map_fn_lst, 'exposure', da_name, idx, 'scale factor'
            ) if scale_factor != 1 else scale_factor

            # Get the weight_factor input parameter.
            da_wf = self.get_param(
                wf_lst, map_fn_lst, 'exposure', da_name, idx, 'weight factor'
            ) if weight_factor != 1 else weight_factor

            # Add the exposure map to config and staticmaps.
            map_name = f'{da_cat}_{da_subcat}' if da_subcat is not None else da_cat
            self.check_uniqueness(
                'exposure',
                map_name,
                {
                    'map_fn': da_map_fn,
                    'cat': da_cat,
                    'subcat': da_subcat,
                    'unit': da_unit,
                    'chunks': chunks if chunks == 'auto' else chunks_lst[idx],
                    'function_fn': da_function_fn,
                    'comp_alg': da_comp_alg,
                    'sf': da_sf,
                    'wf': da_wf,
                },
                file_type='exposure',
                filename=map_name,
            )
            self.update_config(
                'exposure',
                map_name,
                {
                    'map_fn': da_map_fn,
                    'cat': da_cat,
                    'subcat': da_subcat,
                    'unit': da_unit,
                    'chunks': chunks if chunks == 'auto' else chunks_lst[idx],
                    'function_fn': da_function_fn,
                    'comp_alg': da_comp_alg,
                    'sf': da_sf,
                    'wf': da_wf,
                },
            )
            self.set_staticmaps(da, map_name)
            post = f'{da_cat} - {da_subcat}' if da_subcat is not None else da_cat
            self.logger.info(
                f'Added exposure map: {da_name} ({post})'
            )

    def setup_geometry(
        self,
        region,
    ):
        """The area(s) in which the model output is aggregated.

        Adds model layer:

        * **region** geom: A geometry with the nomenclature 'region'.

        Parameters
        ----------
        region: str, Path
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}. See :py:meth:`~hydromt.workflows.parse_region()` for all options.
        """

        # Check the hazard input parameter types.
        if region is not None:
            if not isinstance(region, Path):
                raise TypeError(
                    'The file indicated by the "region" parameter does not exist.'
                )

            # Read the region. # TODO: Add GPKG read, using a spatial index to improve speed!
            geom = io.open_vector(
                region,
                driver='vector',
                dst_crs=self.staticmaps.raster.crs,
                bbox=self.staticmaps.raster.box,
            )

            # Set the model region geometry.
            self.set_staticgeoms(geom, 'region')
            self.logger.info(
                f'Added geometry: region'
            )

    def calculate_damage(
        self,
    ):
        """ """

        # Check (in case of multiple map types) the validity of each combination.
        if len(self.get_config('hazard')) > 1:
            if len(set([len(i) for i in self.get_config('hazard').values()])) != 1:
                raise IndexError(
                    'The number of hazard files per map type are not equal.'
                )
            if any([
                k['rp'] is None for j in [i for i in self.get_config('hazard')]
                for k in self.get_config('hazard', j).values()
            ]):
                raise ValueError(
                    'Not all hazard files are assigned with a return period parameter.'
                )
            if not all([
                k == [
                    [j['rp'] for j in self.get_config('hazard', i).values()] for i in
                    self.get_config('hazard')
                ][0] for k in [
                    [j['rp'] for j in self.get_config('hazard', i).values()] for i in self.get_config('hazard')
                ]
            ]):
                raise ValueError(
                    'The assigned return period parameters per map type are not the same.'
                )

        for idx, hazard_scenario in enumerate(
            zip(*
                [
                    sorted(
                        [(i, j) for j in self.get_config('hazard', i).keys()],
                        key=lambda x: self.get_config('hazard', i, x[1], 'rp'),
                    ) for i in self.get_config('hazard')
                ] if not any(
                    [
                        k['rp'] is None for j in [i for i in self.get_config('hazard')]
                        for k in self.get_config('hazard', j).values()
                    ]
                ) else [
                    [(i, j) for j in self.get_config('hazard', i).keys()] for i in
                    self.get_config('hazard')
                ]
            ),
        ):

            # Calculate and store the damage per exposure category.
            scenario_tag = f'hazard_scenario_{idx + 1}'
            damage_ds = xr.Dataset()
            damage_dict = {}

            if self.get_config('category_output') or self.get_config('total_output'):
                for exposure_fn in self.get_config('exposure'):
                    ds = self.staticmaps.get(
                        [i[1] for i in hazard_scenario] + [exposure_fn],
                    ).unify_chunks()

                    # Read susceptibility function(s).
                    function_fn = self.get_config('exposure', exposure_fn, 'function_fn')
                    function_dict = {}
                    for key in function_fn.keys():
                        keys, values = self.read_susceptibility_function(function_fn[key])
                        function_dict[key] = {
                            'keys': keys,
                            'values': values,
                        }
                        if key not in dict(hazard_scenario).keys():
                            raise ValueError(
                                'The map type to which the susceptibility function is '
                                'assigned does not match with the map types of the '
                                'hazard scenario.'
                            )

                    # Calculate and store the damage.
                    damage_ds[exposure_fn] = xr.map_blocks(
                        self.get_damage,
                        ds,
                        kwargs={
                            'hazard_scenario': dict(hazard_scenario),
                            'exposure_fn': exposure_fn,
                            'function_dict': function_dict,
                            'comp_alg': self.get_config('exposure', exposure_fn, 'comp_alg'),
                            'scale_factor': self.get_config('exposure', exposure_fn, 'sf'),
                            'weight_factor': self.get_config('exposure', exposure_fn, 'wf'),
                        },
                        template=ds[exposure_fn],
                    )
                    damage_dict[exposure_fn] = {
                        'damage': np.nansum(
                            damage_ds[exposure_fn]
                        ),
                        'units_affected': np.nansum(
                            self.staticmaps[exposure_fn].where(
                                damage_ds[exposure_fn] > 0,
                                other=0,
                            )
                        ),
                        'unit': self.get_config('exposure', exposure_fn, 'unit'),
                    }

            # Calculate and store the total damage over all exposure categories.
            if self.get_config('total_output'):
                damage_ds['total'] = damage_ds[list(damage_ds.data_vars)].to_array().sum('variable')
                damage_dict['total'] = np.nansum(
                    damage_ds['total']
                )

            # If a region is specified, determine the damage within the region (and subregions).
            if self.get_config('region'):
                region = self.staticgeoms.get('region')
                for name in damage_ds.data_vars:
                    damage_ds[f'{name}_region'] = damage_ds[name].where(
                        damage_ds[name].raster.geometry_mask(region),
                        other=np.nan,
                    )
                    if name != 'total':
                        damage_dict[f'{name}_region'] = {
                            'damage': np.nansum(
                                damage_ds[f'{name}_region']
                            ),
                            'units_affected': np.nansum(
                                self.staticmaps[name].where(
                                    damage_ds[f'{name}_region'] > 0,
                                    other=0,
                                )
                            ),
                            'unit': self.get_config('exposure', name, 'unit'),
                        }
                    else:
                        damage_dict[f'{name}_region'] = np.nansum(
                                damage_ds[f'{name}_region']
                        )
                    try:
                        id_idx = [
                            i for i, j in enumerate(region.columns) if j.lower() == 'id'
                        ][0]
                    except: #manual handling for Leyte-Samar project
                        id_idx = [
                            i for i, j in enumerate(region.columns) if j.lower() == 'id_2'
                        ][0]
                    if len(region.iloc[:, id_idx]) > 1:
                        print('Start calculating/recording damage for sub-regions.')
                        for idx, id in enumerate(region.iloc[:, id_idx].values):
                            print('Recording/calculating damage for region {}.'.format(id))
                            damage_ds[f'{name}_subregion_{id}'] = damage_ds[name].where(
                                damage_ds[name].raster.geometry_mask(
                                    gpd.GeoDataFrame(region.loc[idx, :].to_frame().T).set_crs(
                                        damage_ds.raster.crs,
                                    ),
                                ),
                                other=np.nan,
                            )
                            if name != 'total':
                                damage_dict[f'{name}_subregion_{id}'] = {
                                    'damage': np.nansum(
                                        damage_ds[f'{name}_subregion_{id}']
                                    ),
                                    'units_affected': np.nansum(
                                        self.staticmaps[name].where(
                                            damage_ds[f'{name}_subregion_{id}'] > 0,
                                            other=0,
                                        )
                                    ),
                                    'unit': self.get_config('exposure', name, 'unit'),
                                }
                            else:
                                damage_dict[f'{name}_subregion_{id}'] = np.nansum(
                                    damage_ds[f'{name}_subregion_{id}']
                                )

            # Add the damage maps to config and staticmaps.
            print('Add the damage maps to config and staticmaps.')
            damage_categories = list(self.get_config('exposure')) + (
                ['total'] if self.get_config('total_output') else []
            )
            for category in damage_categories:
                damage_maps_dict = {
                    ('layer' if key == category else key.replace(category, '').lstrip('_')): damage_dict[key]
                    for key in damage_dict.keys() if category in key
                }
                self.update_config('damage', scenario_tag, category, damage_maps_dict)
                for idx, key in enumerate(damage_maps_dict):
                    self.set_staticmaps(
                        damage_ds[f'{category}{("_" + key if key != "layer" else "")}'],
                        f'{scenario_tag}_{category}_damage_{key}',
                    )
                    map_name = f'{scenario_tag}_{category}_damage'
                    post = 'layer' if idx == 0 else key.replace("_", " ")
                    self.logger.info(
                        f'Added damage map: {map_name} ({post})'
                    )

    def calculate_risk(
        self,
    ):
        """ """

        # Check if the risk calculation should be conducted.
        if not self.get_config('risk_output'):
            return

        risk_ds = xr.Dataset()
        risk_dict = {}

        # Calculate the risk and store for each exposure category.
        if self.get_config('category_output') or self.get_config('total_output'):
            for exposure_fn in self.get_config('exposure'):
                ds = xr.Dataset()
                for hazard_scenario in self.get_config('damage').keys():
                    damage_fn = f'{hazard_scenario}_{exposure_fn}_damage_layer'
                    ds[damage_fn] = self.staticmaps.get(damage_fn)

                # Get the return periods associated to the hazard scenario.
                rp_lst = [
                    [j['rp'] for j in self.get_config('hazard', i).values()] for
                    i in self.get_config('hazard')
                ][0]
                if not len(set(rp_lst)) == len(rp_lst):
                    raise ValueError(
                        'The return periods of the associated hazard scenario are not unique.'
                    )

                # Determine the coefficients to conduct the log-linear integration.
                cov_lst = self.get_coefficients(rp_lst)

                # Calculate the risk.
                risk_ds[exposure_fn] = xr.map_blocks(
                    self.get_risk,
                    ds,
                    kwargs={
                        'cov_lst': cov_lst,
                    },
                    template=ds[damage_fn],
                )
                risk_dict[exposure_fn] = np.nansum(
                    risk_ds[exposure_fn]
                )

        # Calculate and store the total risk over all exposure categories. # TODO: Check if calculation is correct!
        if self.get_config('total_output'):
            risk_ds['total'] = risk_ds[list(risk_ds.data_vars)].to_array().sum('variable')
            risk_dict['total'] = np.nansum(
                risk_ds['total']
            )

        # If a region is specified, determine the risk within the region (and subregions).
        if self.get_config('region'):
            region = self.staticgeoms.get('region')
            for name in risk_ds.data_vars:
                risk_ds[f'{name}_region'] = risk_ds[name].where(
                    risk_ds[name].raster.geometry_mask(region),
                    other=0,
                )
                risk_dict[f'{name}_region'] = np.nansum(
                    risk_ds[f'{name}_region']
                )
                id_idx = [
                    i for i, j in enumerate(region.columns) if j.lower() == 'id'
                ][0]
                if len(region.iloc[:, id_idx]) > 1:
                    for idx, id in enumerate(region.iloc[:, id_idx].values):
                        risk_ds[f'{name}_subregion_{id}'] = risk_ds[name].where(
                            risk_ds[name].raster.geometry_mask(
                                gpd.GeoDataFrame(region.loc[idx, :].to_frame().T).set_crs(
                                    risk_ds.raster.crs,
                                ),
                            ),
                            other=0,
                        )
                        risk_dict[f'{name}_subregion_{id}'] = np.nansum(
                            risk_ds[f'{name}_subregion_{id}']
                        )

        # Add the risk maps to config and staticmaps.
        risk_categories = list(self.get_config('exposure')) + (
            ['total'] if self.get_config('total_output') else []
        )
        for category in risk_categories:
            risk_maps_dict = {
                ('layer' if key == category else key.replace(category, '').lstrip('_')):
                    risk_dict[key] for key in risk_dict.keys() if category in key
            }
            self.update_config('risk', category, risk_maps_dict)
            for idx, key in enumerate(risk_maps_dict):
                self.set_staticmaps(
                    risk_ds[f'{category}{("_" + key if key != "layer" else "")}'],
                    f'{category}_risk_{key}',
                )
                map_name = f'{category}_risk'
                post = 'layer' if idx == 0 else key.replace("_", " ")
                self.logger.debug(
                    f'Added risk map: {map_name} ({post})'
                )

    def write_results(
        self,
    ):
        """ """

        # Remove historical results from the output folder if present.
        dst_dir = self.get_config('output_dp')
        if any(dst_dir.iterdir()):
            for file in dst_dir.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)

        # Check if a correct combination of output indicators is parsed.
        if not self.get_config('category_output') and not self.get_config('total_output'):
            raise ValueError(
                f'At least one of the "category_output" or "total_output" indicators '
                f'must be enabled in order to create an impact report.'
            )

        # Initiate the writer and output formats of the impact report.
        writer = pd.ExcelWriter(
            self.get_config('output_dp').joinpath('impact_report.xlsx'),
            engine='xlsxwriter',
        )
        wb = writer.book
        header_format = wb.add_format(
            {
                'bold': True,
                'border': 1,
            },
        )
        value_format = wb.add_format(
            {
                'bold': False,
                'align': 'right',
                'border': 1,
            },
        )

        # Store the config information to the impact report.
        config_df = pd.DataFrame(
            [
                ['Date', datetime.date.today().strftime('%d-%m-%Y')],
                ['FIAT version', __version__],
                ['Strategy', self.get_config('strategy')],
                ['Scenario', self.get_config('scenario')],
                ['Year', self.get_config('year')],
                ['Hazard Type', self.get_config('hazard_type')],
            ],
            columns=['Key', 'Value']
        )
        config_df.to_excel(
            writer,
            sheet_name='Impact Report',
            header=False,
            index=False,
        )
        for row_idx in range(len(config_df)):
            writer.sheets['Impact Report'].write(
                row_idx,
                0,
                config_df.iloc[row_idx, 0],
                header_format,
            )
            writer.sheets['Impact Report'].write(
                row_idx,
                1,
                config_df.iloc[row_idx, 1],
                value_format,
            )

        # Get the damage results.
        damage_df = pd.DataFrame()
        categories = list(set(
            [self.get_config('exposure', i, 'cat') for i in self.get_config('exposure')]
        )) + ['total']
        subcategories = list(set([
            self.get_config('exposure', i, 'subcat') for i in
            self.get_config('exposure')
        ]))
        for hazard_idx, hazard_scenario in enumerate(
                zip(*
                    [
                        sorted(
                            [(i, j) for j in self.get_config('hazard', i).keys()],
                            key=lambda x: self.get_config('hazard', i, x[1], 'rp'),
                        ) for i in self.get_config('hazard')
                    ] if not any(
                        [
                            k['rp'] is None for j in
                            [i for i in self.get_config('hazard')]
                            for k in self.get_config('hazard', j).values()
                        ]
                    ) else [
                        [(i, j) for j in self.get_config('hazard', i).keys()] for i in
                        self.get_config('hazard')
                    ]
                    ),
        ):

            # Determine the scenario tag and name.
            scenario_tag = f'hazard_scenario_{hazard_idx + 1}'
            if len(hazard_scenario) == 1:
                if all([i['rp'] is not None for i in self.get_config('hazard', hazard_scenario[0][0]).values()]):
                    scenario_name = scenario_tag + f' [RP{self.get_config("hazard", hazard_scenario[0][0], hazard_scenario[0][1], "rp")}]'
                else:
                    scenario_name = scenario_tag
            else:
                scenario_name = scenario_tag + f' [RP{np.unique([self.get_config("hazard", i[0], i[1], "rp") for i in hazard_scenario])[0]}]'

            if not self.get_config('category_output') and self.get_config('total_output'):
                damage_maps = [
                    [i, j] for i in self.get_config('damage', scenario_tag) for j in
                    self.get_config('damage', scenario_tag, i).keys() if 'total' in i
                ]
            else:
                damage_maps = [
                    [i, j] for i in self.get_config('damage', scenario_tag) for j in
                    self.get_config('damage', scenario_tag, i).keys()
                ]
            damage_df['Category'] = [
                j for i in damage_maps for j in categories if j in i[0]
            ]
            damage_df['Subcategory'] = [
                [subcategories[j] for j, k in enumerate(subcategories) if k in i[0]][0]
                if any([j in i[0] for j in subcategories if j is not None]) else '-'
                for i in damage_maps
            ]
            damage_df['Domain'] = [
                i[1].replace('_', ' ') for i in damage_maps
            ]
            damage_df[scenario_name, f'Damage [{self.get_config("output_unit")}]'] = [
                int(self.get_config('damage', scenario_tag, i[0], i[1], 'damage'))
                if not 'total' in i[0] else
                int(self.get_config('damage', scenario_tag, i[0], i[1]))
                for i in damage_maps
            ]
            damage_df[scenario_name, 'Units Affected'] = [
                int(self.get_config('damage', scenario_tag, i[0], i[1], 'units_affected'))
                if not 'total' in i[0] else '-' for i in damage_maps
            ]
            damage_df[scenario_name, 'Unit'] = [
                self.get_config('damage', scenario_tag, i[0], i[1], 'unit')
                if not 'total' in i[0] else '-' for i in damage_maps
            ]
            # Store the damage maps.
            if self.get_config('map_output'):
                dst_dir.joinpath(scenario_tag).mkdir(parents=True, exist_ok=True)
                for map in damage_maps:
                    map_name = f'{scenario_tag}_{map[0]}_damage_{map[1]}'
                    self.staticmaps[map_name].raster.to_raster(
                        dst_dir.joinpath(scenario_tag, f'{map_name}.tif'),
                        compress='lzw',
                        nodata=0,
                        logger=self.logger,
                    )
                    self.logger.info(
                        f'Stored damage map: {scenario_tag}_{map[0]}_damage ({map[1].replace("_", " ")})'
                    )

            # Store the damage results to the region geometry.
            if self.get_config('region') and self.get_config('map_output'):
                region = self.staticgeoms.get('region')
                id_idx = [
                    i for i, j in enumerate(region.columns) if j.lower() == 'id'
                ][0]
                for row in damage_df.iterrows():
                    for idx, id in enumerate(region.iloc[:, id_idx].values):
                        domain_tag = 'region' if len(region.iloc[:, id_idx]) == 1 else f'subregion {id}'
                        if row[1]['Domain'] == domain_tag:
                            if row[1]['Category'] != 'total':
                                damage_name = f'{scenario_tag}_{row[1]["Category"]}_{row[1]["Subcategory"]}'
                                self.staticgeoms['region'].loc[
                                    idx,
                                    f'{damage_name}_damage',
                                ] = row[1][3 + 3*hazard_idx]
                                self.staticgeoms['region'].loc[
                                    idx,
                                    f'{damage_name}_units_affected',
                                ] = row[1][4 + 3*hazard_idx]
                                self.staticgeoms['region'].loc[
                                    idx,
                                    f'{damage_name}_unit',
                                ] = row[1][5 + 3*hazard_idx]
                            else:
                                damage_name = f'{scenario_tag}_{row[1]["Category"]}'
                                self.staticgeoms['region'].loc[
                                    idx,
                                    f'{damage_name}_damage',
                                ] = row[1][3 + 3*hazard_idx]
                            self.logger.info(
                                f'Updated region geometry: {damage_name}_damage ({domain_tag})'
                            )

        # Store the damage results to the impact report.
        damage_df.set_index(['Category', 'Subcategory', 'Domain'], drop=True, inplace=True)
        damage_df.to_excel(
            writer,
            sheet_name='Impact Report',
            header=False,
            startrow=8,
        )
        for row_idx in range(len(damage_df)):
            for col_idx in range(len(damage_df.columns)):
                writer.sheets['Impact Report'].write(
                    8 + row_idx,
                    3 + col_idx,
                    damage_df.iloc[row_idx,col_idx],
                    value_format,
                )

        # Get the risk results.
        if self.get_config('risk_output'):
            risk_df = pd.DataFrame()
            if not self.get_config('category_output') and self.get_config('total_output'):
                risk_maps = [
                    [i, j] for i in self.get_config('risk') for j in
                    self.get_config('risk', i).keys() if 'total' in i
                ]
            else:
                risk_maps = [
                    [i, j] for i in self.get_config('risk') for j in
                    self.get_config('risk', i).keys()
                ]
            risk_df[f'Risk [{self.get_config("output_unit")}]'] = [
                int(self.get_config('risk', i[0], i[1])) for i in risk_maps
            ]

            # Store the risk maps.
            if self.get_config('map_output'):
                dst_dir.joinpath('risk').mkdir(parents=True, exist_ok=True)
                for map in risk_maps:
                    map_name = f'{map[0]}_risk_{map[1]}'
                    self.staticmaps[map_name].raster.to_raster(
                        dst_dir.joinpath('risk', f'{map_name}.tif'),
                        compress='lzw',
                        nodata=0,
                        logger=self.logger,
                    )
                    self.logger.info(
                        f'Stored risk map: {map[0]}_risk ({map[1].replace("_", " ")})'
                    )

            # Store the risk results to the region geometry.
            if self.get_config('region') and self.get_config('map_output'):
                region = self.staticgeoms.get('region')
                id_idx = [
                    i for i, j in enumerate(region.columns) if j.lower() == 'id'
                ][0]
                for row in pd.concat(
                        [damage_df.reset_index().iloc[:, :3], risk_df],
                        axis=1,
                ).iterrows():
                    for idx, id in enumerate(region.iloc[:, id_idx].values):
                        domain_tag = 'region' if len(region.iloc[:, id_idx]) == 1 else f'subregion {id}'
                        if row[1]['Domain'] == domain_tag:
                            if row[1]['Category'] != 'total':
                                risk_name = f'{row[1]["Category"]}_{row[1]["Subcategory"]}'
                            else:
                                risk_name = row[1]['Category']
                            self.staticgeoms['region'].loc[
                                idx,
                                f'{risk_name}_risk',
                            ] = row[1][3]
                            self.logger.info(
                                f'Updated region geometry: {risk_name}_risk ({domain_tag})'
                            )

            # Store the risk results to the impact report.
            risk_df.to_excel(
                writer,
                sheet_name='Impact Report',
                index=False,
                startrow=7,
                startcol=3 + 3*len(self.get_config('damage')),
            )
            writer.sheets['Impact Report'].write(6, 3 + len(damage_df.columns), '', header_format)
            for row_idx in range(len(risk_df)):
                writer.sheets['Impact Report'].write(8 + row_idx, 3 + len(damage_df.columns), risk_df.iloc[row_idx, 0], value_format)

        # Set the header of the impact report.
        header_df = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(damage_df.columns)
        )
        header_df.to_excel(
            writer,
            sheet_name='Impact Report',
            startrow=6,
            startcol=2
        )

        # Set the index of the impact report.
        index_df = pd.DataFrame(
            {
                'Category': [],
                'Subcategory': [],
                'Domain': [],
            }
        )
        index_df.to_excel(
            writer,
            sheet_name='Impact Report',
            index=False,
            startrow=7,
        )

        # Merge celss A7:C7 of the impact report.
        writer.sheets['Impact Report'].merge_range('A7:C7', '', header_format)

        # Save the import report.
        writer.save()
        self.logger.info(
            f'Stored the impact report.'
        )

        # Save the region geometry.
        if self.get_config('region'):
            self.staticgeoms.get('region').to_file(
                self.get_config('output_dp').joinpath('regional_impact.geojson'),
                driver='GeoJSON',
            )
            self.logger.info(
                f'Stored the regional impact results.'
            )

    """ SUPPORT FUNCTIONS """

    def get_area_grid(self, ds):
        """Returns a xarray.DataArray containing the area in [m2] of the reference grid ds.

        Parameters
        ----------
        ds : xarray.DataArray or xarray.DataSet
            xarray.DataArray or xarray.DataSet containing the reference grid(s).

        Returns
        -------
        da_area : xarray.DataArray
            xarray.DataArray containing the area in [m2] of the reference grid.
        """

        if ds.raster.crs.is_geographic:
            area = gis_utils.reggrid_area(
                ds.raster.ycoords.values,
                ds.raster.xcoords.values,
            )
            da_area = xr.DataArray(
                data=area.astype('float32'),
                coords=ds.raster.coords,
                dims=ds.raster.dims,
            )

        elif ds.raster.crs.is_projected:
            da = ds[list(ds.data_vars)[0]] if isinstance(ds, xr.Dataset) else ds
            xres = abs(da.raster.res[0]) * da.raster.crs.linear_units_factor[1]
            yres = abs(da.raster.res[1]) * da.raster.crs.linear_units_factor[1]
            da_area = xr.full_like(da, fill_value=1, dtype=np.float32) * xres * yres

        da_area.raster.set_nodata(0)
        da_area.raster.set_crs(ds.raster.crs)
        da_area.attrs.update(unit='m2')

        return da_area.rename('area')

    def get_coefficients(self, rp_lst):
        """ """

        f_lst = [1 / i for i in rp_lst]
        lf = [np.log(1 / i) for i in rp_lst]
        c = [(1 / (lf[i] - lf[i+1])) for i in range(len(rp_lst[:-1]))]
        G = [(f_lst[i] * lf[i] - f_lst[i]) for i in range(len(rp_lst))]
        a = [((1 + c[i] * lf[i+1]) * (f_lst[i] - f_lst[i+1]) + c[i] * (G[i+1] - G[i])) for i in range(len(rp_lst[:-1]))]
        b = [(c[i] * (G[i] - G[i+1] + lf[i+1] * (f_lst[i+1] - f_lst[i]))) for i in range(len(rp_lst[:-1]))]
        if len(rp_lst) == 1:
            cov_lst = f_lst
        else:
            cov_lst = [b[0] if i == 0 else f_lst[i] + a[i-1] if i == len(rp_lst) - 1 else a[i-1] + b[i] for i in range(len(rp_lst))]

        return cov_lst

    def get_config(self, *args, fallback=None, abs_path=False):
        """Get a config value at key(s).

        Parameters
        ----------
        args : tuple or string
            Keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        fallback: any, optional
            Fallback value if key(s) not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root, by deafult False.
            NOTE: this assumes the config is located in model root!

        Returns
        value : any type
            Dictionary value

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> get_config('a')
        >> 1

        >> get_config('b', 'c', 'd') # identical to get_config('b.c.d')
        >> 2

        >> get_config('b.c') # # identical to get_config('b','c')
        >> {'d': 2}
        """
        args = list(args)
        if len(args) == 1 and '.' in args[0]:
            args = args[0].split('.') + args[1:]
        branch = self.config
        for key in args[:-1]:
            branch = branch.get(key, {})
            if not isinstance(branch, dict):
                branch = dict()
                break

        value = branch.get(args[-1], fallback)
        if abs_path and isinstance(value, str):
            value = Path(value)
            if not value.is_absolute():
                value = self.root.joinpath(value)

        return value

    def get_damage(self, ds, hazard_scenario, exposure_fn, function_dict, comp_alg, scale_factor, weight_factor):
        """ """

        damage_factors = []

        for key in function_dict:
            # Check if the hazard values exceed the minimum bound of the susceptibility function.
            if np.nanmin(ds[hazard_scenario[key]].values) < function_dict[key]['keys'][0]:
                raise IndexError(
                    f'The hazard values exceed the minimum bound of the susceptibility function.'
                )

            # Check and resolve if the hazard values exceed the maximum bound of the susceptibility function.
            if np.nanmax(ds[hazard_scenario[key]].values) > function_dict[key]['keys'][-1]:
                ds[hazard_scenario[key]] = ds[hazard_scenario[key]].where(
                    ~(ds[hazard_scenario[key]].values > function_dict[key]['keys'][-1]),
                    other=function_dict[key]['keys'][-1],
                )

            # Determine the damage factors given the hazard intensity values.
            index = np.digitize(
                ds[hazard_scenario[key]],
                np.append(function_dict[key]['keys'], np.nan),
                right=True,
            )
            damage_factors.append(np.append(function_dict[key]['values'], np.nan)[index])

        # Calculate the compounded damage factors.
        if comp_alg == 'average':
            damage_factors = np.nanmean(damage_factors, axis=0)
        elif comp_alg == 'max':
            damage_factors = np.nanmax(damage_factors, axis=0)

        damage_factors = np.where(
            damage_factors != damage_factors,
            0,
            damage_factors
        )

        return ds[exposure_fn] * damage_factors * scale_factor * weight_factor

    def get_density_grid(self, ds):
        """Returns a xarray.DataArray or DataSet containing the density in [unit/m2] of the reference grid(s) ds.

        Parameters
        ----------
        ds: xarray.DataArray or xarray.DataSet
            xarray.DataArray or xarray.DataSet containing reference grid(s).

        Returns
        -------
        ds_out: xarray.DataArray or xarray.DataSet
            xarray.DataArray or xarray.DataSet containing the density in [unit/m2] of the reference grid(s).
        """

        # Create a grid that contains the area in m2 per grid cell.
        if ds.raster.crs.is_geographic:
            area = self.get_area_grid(ds)

        elif ds.raster.crs.is_projected:
            xres = abs(ds.raster.res[0]) * ds.raster.crs.linear_units_factor[1]
            yres = abs(ds.raster.res[1]) * ds.raster.crs.linear_units_factor[1]
            area = xres * yres

        # Create a grid that contains the density in unit/m2 per grid cell.
        ds_out = ds / area

        return ds_out

    def get_risk(self, ds, cov_lst):
        """ """

        # Determine the risk given the log-linear integration coefficient values.
        da_risk = xr.Dataset()
        for i, name in enumerate(ds.data_vars):
            da = ds[name].where(ds[name] == ds[name], other=0)
            if i == 0:
                da_risk = da * cov_lst[i]
            else:
                da_risk += da * cov_lst[i]

        return da_risk

    def get_param(self, param_lst, map_fn_lst, file_type, filename, i, param_name):
        """ """

        if len(param_lst) == 1:
            return param_lst[0]
        elif len(param_lst) != 1 and len(map_fn_lst) == len(param_lst):
            return param_lst[i]
        elif len(param_lst) != 1 and len(map_fn_lst) != len(param_lst):
            raise IndexError(
                f'Could not derive the {param_name} parameter for {file_type} map: {filename}.'
            )

    def read_susceptibility_function(self, function_fn):
        """ """

        # Read the vulnerability function.
        if function_fn.suffix == '.csv':
            df_function = pd.read_csv(function_fn).dropna()
        df_function.columns = ['x', 'y']

        # Check if hazard intensity values (x-values) are unique and ascending.
        try:
            assert df_function.loc[:, 'x'].diff().min() > 0
        except AssertionError:
            raise ValueError(
                f'The requirement that hazard intensity values of the vulnerability function are unique and ascending is not met.'
            )

        # Interpolate the vulnerability function.
        keys = np.arange(
            start=int(df_function.loc[:, 'x'].min() * 100),
            stop=int(df_function.loc[:, 'x'].max() * 100) + 1,
            step=1,
        ) * 0.01
        values = np.interp(
            x=keys,
            xp=df_function.loc[:, 'x'].values,
            fp=df_function.loc[:, 'y'].values,
        )

        return keys, values

    def run_log_method(self, method, *args, **kwargs):
        """Log method parameters before running a method."""

        method = method.strip('0123456789')
        func = getattr(self, method)
        signature = inspect.signature(func)
        for i, (k, v) in enumerate(signature.parameters.items()):
            v = kwargs.get(k, v.default)
            if v is inspect.Parameter.empty:
                if len(args) >= i + 1:
                    v = args[i]
                else:
                    continue
            self.logger.info(
                f'{method}.{k}: {v}'
            )

        return func(*args, **kwargs)

    def set_root(self, root):
        """Initialized the model root."""

        # If not specified, get root from configuration file location.
        if root is None:
            root = Path(self.config_fn).parent

        # Check if directory exist.
        if not Path(root).is_dir():
            raise IOError(
                f'Model root not found at "{root}".'
            )
        self.root = Path(root)

    def set_staticgeoms(self, geom, name):
        """Add geom to staticmaps"""
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(geom, t) for t in gtypes]):
            raise TypeError(
                'First parameter map(s) should be geopandas.GeoDataFrame or geopandas.GeoSeries.'
            )
        if name in self.staticgeoms:
            self.logger.warning(
                f'Overwriting staticgeom: {name}'
            )
        self.staticgeoms[name] = geom

    def set_staticmaps(self, data, name=None):
        """Add data to staticmaps.

        All layers of staticmaps must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to staticmaps
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xr.DataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xr.Dataset):
                raise ValueError(
                    'Setting a map requires a name.'
                )
        elif name is not None and isinstance(data, xr.Dataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError(
                    'Name not found in DataSet.'
                )
            else:
                data = data[[name]]
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()
        if len(self.staticmaps) == 0:
            self.staticmaps = data
        else:
            if isinstance(data, np.ndarray):
                if data.shape != self.shape:
                    raise ValueError(
                        'Shape of data and staticmaps do not match.'
                    )
                data = xr.DataArray(dims=self.dims, data=data, name=name).to_dataset()
            for var in data.data_vars.keys():
                if var in self.staticmaps:
                    self.logger.warning(
                        f'Overwriting staticmap: {var}'
                    )
                self.staticmaps[var] = data[var]

    def update_config(self, *args):
        """Update the config dictionary at key(s) with values."""

        if len(args) < 2:
            raise TypeError(
                'Method update_config() requires a least one key and one value.'
            )
        args = list(args)
        value = args.pop(-1)
        if len(args) == 1 and '.' in args[0]:
            args = args[0].split('.') + args[1:]
        branch = self.config
        for key in args[:-1]:
            if not key in branch or not isinstance(branch[key], dict):
                branch[key] = {}
            branch = branch[key]

        branch[args[-1]] = value

    """ CONTROL FUNCTIONS """

    def check_dir_exist(self, dir, name=None):
        """ """

        if not isinstance(dir, Path):
            raise TypeError(
                f'The directory indicated by the "{name}" parameter does not exist.'
            )

    def check_file_exist(self, param_lst, name=None, input_dir=None):
        """ """

        for param_idx, param in enumerate(param_lst):
            if isinstance(param, dict):
                fn_lst = list(param.values())
            else:
                fn_lst = [param]
            for fn_idx, fn in enumerate(fn_lst):
                if not Path(fn).is_file():
                    if self.root.joinpath(fn).is_file():
                        if isinstance(param, dict):
                            param_lst[param_idx][list(param.keys())[fn_idx]] = self.root.joinpath(fn)
                        else:
                            param_lst[param_idx] = self.root.joinpath(fn)
                    if input_dir is not None:
                        if self.get_config(input_dir).joinpath(fn).is_file():
                            if isinstance(param, dict):
                                param_lst[param_idx][list(param.keys())[fn_idx]] = self.get_config(input_dir).joinpath(fn)
                            else:
                                param_lst[param_idx] = self.get_config(input_dir).joinpath(fn)
                else:
                    if isinstance(param, dict):
                        param_lst[param_idx][list(param.keys())[fn_idx]] = Path(fn)
                    else:
                        param_lst[param_idx] = Path(fn)
                try:
                    if isinstance(param, dict):
                        assert isinstance(param_lst[param_idx][list(param.keys())[fn_idx]], Path) == True
                    else:
                        assert isinstance(param_lst[param_idx], Path) == True
                except AssertionError:
                    if input_dir is None:
                        raise TypeError(
                            f'The file indicated by the "{name}" parameter does not'
                            f' exist in the directory "{self.root}".'
                        )
                    else:
                        raise TypeError(
                            f'The file indicated by the "{name}" parameter does not'
                            f' exist in either of the directories "{self.root}" or '
                            f'"{self.get_config(input_dir)}".'
                        )

    def check_get_opt(self, opt):
        """Check all opt keys and raise sensible error messages if unknown."""

        for method in opt.keys():
            m = method.strip('0123456789')
            if not callable(getattr(self, m, None)):
                if not hasattr(self, m) and hasattr(self, f'setup_{m}'):
                    raise DeprecationWarning(
                        f'Use full name "setup_{method}" instead of "{method}"'
                    )
                else:
                    raise ValueError(
                        f'FiatModel has no method "{method}"'
                    )

        return opt

    def check_param_type(self, param, name=None, types=None):
        """ """

        if not isinstance(param, list):
            raise TypeError(
                f'The "{name}_lst" parameter should be a of {list}, received a '
                f'{type(param)} instead.'
            )
        for i in param:
            if not isinstance(i, types):
                if isinstance(types, tuple):
                    types = ' or '.join([str(j) for j in types])
                else:
                    types = types
                raise TypeError(
                    f'The "{name}" parameter should be a of {types}, received a '
                    f'{type(i)} instead.'
                )

    def check_uniqueness(self, *args, file_type=None, filename=None):
        """ """

        args = list(args)
        if len(args) == 1 and '.' in args[0]:
            args = args[0].split('.') + args[1:]
        branch = args.pop(-1)
        for key in args[::-1]:
            branch = {key: branch}

        if self.get_config(args[0], args[1]):
            for key in self.staticmaps.data_vars:
                if filename == key:
                    raise ValueError(
                        f'The filenames of the {file_type} maps should be unique.'
                    )
                if self.get_config(args[0], args[1], key) == list(branch[args[0]][args[1]].values())[0]:
                    raise ValueError(
                        f"Each model input layers must be unique."
                    )


# Parse arguments.
@click.command()
@click.option(
        '-i',
        '--config',
        type=click.Path(resolve_path=True),
        help='Path to FIAT configuration file.',
    )


def main(config):

    # Set root.
    root = Path(config).parent

    # Set up logger.
    logger = setuplog(
        'run',
        root.joinpath('fiat.log'),
        log_level='DEBUG',
        append=False
    )

    # Run the FIAT model.
    model = FiatModel(
        root=root,
        config_fn=config,
        logger=logger,
    )
    model.run()


if __name__ == '__main__':
    main()