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
    assert sorted(os.listdir(os.path.dirname(item))) == ['l2b.json', 'l2b.log', 'l2b.nc']


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


def test_item_reads_with_pystac(item):
    it = pystac.Item.from_file(item)
    assert it.assets['data'].get_absolute_href() == os.path.join(os.path.dirname(item), 'l2b.nc')


def test_cli_product_dir_and_stac(tmp_path, l2a_path):
    run = [sys.executable, '-c', 'from GRSl2bgen.run import main; main()',
           l2a_path, '--odir', str(tmp_path), '--product_dir', '--stac', '--no_clobber']
    subprocess.run(run, check=True, capture_output=True)
    name = 'S2A_MSIL2B_20210211_T31TFJ_camargue_subset'
    assert sorted(os.listdir(tmp_path / name)) == [name + ext for ext in ('.json', '.log', '.nc')]
    # complete product (its Item exists): skipped
    out = subprocess.run(run, check=True, capture_output=True, text=True).stdout
    assert 'already processed; skip!' in out
