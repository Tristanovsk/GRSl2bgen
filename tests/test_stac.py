"""STAC Item of an L2B product (GRSl2bgen.stac)."""

import json
import os
import shutil
import subprocess
import sys

import pystac
import pytest

from GRSl2bgen.stac import item_path, write_stac_item


@pytest.fixture(scope='module')
def item(tmp_path_factory, l2b_nc, l2a_path):
    """Product folder <name>/ with the L2B, a log and the Item."""
    odir = tmp_path_factory.mktemp('stac') / 'l2b'
    odir.mkdir()
    ofile = odir / 'l2b.nc'
    shutil.copy(l2b_nc, ofile)
    (odir / 'l2b.log').write_text('log of the run\n')
    path = write_stac_item(ofile, l2a=l2a_path, log_file=odir / 'l2b.log')
    return path


def test_item_next_to_product(item):
    assert item == item_path(os.path.join(os.path.dirname(item), 'l2b.nc'))
    assert sorted(os.listdir(os.path.dirname(item))) == ['l2b.json', 'l2b.log', 'l2b.nc',
                                                         'l2b_ql_l2b.png']


def test_item_content(item, l2a_path):
    d = json.load(open(item))
    assert d['id'] == 'l2b'
    assert d['assets']['data']['href'] == './l2b.nc'
    assert d['assets']['data']['roles'] == ['data']
    assert d['assets']['log'] == {**d['assets']['log'], 'href': './l2b.log', 'roles': ['metadata']}
    assert d['assets']['log']['file:size'] == len('log of the run\n')
    assert d['assets']['data']['file:checksum'].startswith('1220')
    p = d['properties']
    assert p['proj:code'] == 'EPSG:32631' and p['proj:shape'] == [128, 128]
    assert p['processing:level'] == 'L2B' and 'GRSl2bgen' in p['processing:software']
    assert p['grid:code'] == 'MGRS-31TFJ'
    assert p['datetime'].startswith('2021-02-11T10:41:51')
    assert [link['rel'] for link in d['links']] == ['derived_from']
    # footprint of the 128 x 128 crop in Camargue
    lon0, lat0, lon1, lat1 = d['bbox']
    assert 4.4 < lon0 < lon1 < 4.6 and 43.4 < lat0 < lat1 < 43.6


def test_derived_from_local_path(item, l2a_path):
    d = json.load(open(item))
    assert d['links'][0]['href'] == os.path.abspath(str(l2a_path).rstrip('/'))


def test_derived_from_url_kept(tmp_path, l2b_nc):
    """An L2A product in a bucket is linked by its URL, not turned into a local path."""
    ofile = tmp_path / 'l2b.nc'
    shutil.copy(l2b_nc, ofile)
    url = 's3://bucket/L2A_zarr/31TFJ/2021/02/11/product/product.zarr'
    d = json.load(open(write_stac_item(ofile, l2a=url + '/')))
    assert [link['href'] for link in d['links'] if link['rel'] == 'derived_from'] == [url]
    assert d['properties']['processing:lineage'].endswith(' product.zarr')


def test_item_reads_with_pystac(item):
    it = pystac.Item.from_file(item)
    assert it.assets['data'].get_absolute_href() == os.path.join(os.path.dirname(item), 'l2b.nc')


def test_cli_product_dir_and_stac(tmp_path, l2a_path):
    run = [sys.executable, '-c', 'from GRSl2bgen.run import main; main()',
           l2a_path, '--odir', str(tmp_path), '--product_dir', '--stac', '--no_clobber']
    subprocess.run(run, check=True, capture_output=True)
    name = 'S2A_MSIL2B_20210211_T31TFJ_camargue_subset'
    assert sorted(os.listdir(tmp_path / name)) == [name + ext for ext in
                                                   ('.json', '.log', '.nc', '_ql_l2b.png')]
    # complete product (its Item exists): skipped
    out = subprocess.run(run, check=True, capture_output=True, text=True).stdout
    assert 'already processed; skip!' in out


def test_quicklook_asset(item):
    """The quick look is written next to the product and is the thumbnail asset of the Item."""
    d = json.load(open(item))
    thumbnail = d['assets']['thumbnail']
    assert thumbnail['href'] == './l2b_ql_l2b.png' and thumbnail['roles'] == ['thumbnail']
    assert thumbnail['type'] == 'image/png' and thumbnail['file:checksum'].startswith('1220')
    png = os.path.join(os.path.dirname(item), 'l2b_ql_l2b.png')
    with open(png, 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'
    assert thumbnail['file:size'] == os.path.getsize(png)


def test_quicklook_of_pyramid(tmp_path, l2b_pyramid):
    """A Zarr pyramid is read at its finest level not larger than max_size."""
    from GRSl2bgen.quicklook import open_coarse, write_quicklook

    with open_coarse(l2b_pyramid, max_size=64) as ds:
        assert max(ds.sizes['x'], ds.sizes['y']) <= 64
    with open_coarse(l2b_pyramid, max_size=1000) as ds:
        assert ds.sizes['x'] == 128
    png = write_quicklook(l2b_pyramid, path=tmp_path / 'ql.png', max_size=64)
    assert png == str(tmp_path / 'ql.png') and os.path.getsize(png) > 0


def test_item_without_quicklook(tmp_path, l2b_nc):
    """quicklook=False: no image and no thumbnail asset."""
    ofile = tmp_path / 'l2b.nc'
    shutil.copy(l2b_nc, ofile)
    d = json.load(open(write_stac_item(ofile, quicklook=False)))
    assert 'thumbnail' not in d['assets']
    assert sorted(os.listdir(tmp_path)) == ['l2b.json', 'l2b.nc']


def test_failed_quicklook_keeps_item(tmp_path, l2b_nc, monkeypatch, caplog):
    """A quick look that cannot be written leaves the Item without thumbnail, with a warning."""
    import GRSl2bgen.stac as stac

    def fail(ofile):
        raise RuntimeError('no display')

    monkeypatch.setattr(stac, 'write_quicklook', fail)
    ofile = tmp_path / 'l2b.nc'
    shutil.copy(l2b_nc, ofile)
    d = json.load(open(stac.write_stac_item(ofile)))
    assert 'thumbnail' not in d['assets']
    assert 'no quick look' in caplog.text and 'no display' in caplog.text


def test_quicklook_without_parameters(tmp_path, l2b_nc, caplog):
    """A product without the parameters of the quick look gives no image, with a warning."""
    import xarray as xr

    from GRSl2bgen.quicklook import PANELS, write_quicklook

    ofile = tmp_path / 'other.nc'
    with xr.open_dataset(l2b_nc) as ds:
        ds.drop_vars([p[0] for p in PANELS]).to_netcdf(ofile)
    assert write_quicklook(ofile) is None
    assert not os.path.exists(tmp_path / 'other_ql_l2b.png')
    assert 'no quick look' in caplog.text
