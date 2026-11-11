"""Hash-pinned Natural Earth country boundaries and conservative exact aliases."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'ne-map-units.zip': ('https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_map_units.zip','45bebe2aaf8bf42b9daf4428594925bcb11eabd32fe9dc6e0acf681438053eb5'),
    'natural-earth.zip': ('https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip', 'ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6'),
    'ne-admin1.zip': ('https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip', 'efc59726337323058f9446210adc96673179cd344e053666ee3d28cb58ba2b05'),
}


def prepare(folder):
    # pyshp is only a local source-preparation dependency; never installed remotely.
    sys.path.insert(0, str(ROOT / 'target/minds-we-choose/deps'))
    import shapefile
    sources = []
    aliases = {}
    country_aliases = {}
    territories = {'French Guiana','Martinique','Guadeloupe','Reunion','Mayotte','Cocos (Keeling) Islands','Christmas Island','Svalbard','Jan Mayen','Bouvet Island'}
    def add(key, name):
        if key and key != '-99':
            aliases.setdefault(key.strip().casefold(), set()).add(name)
    geometry = []
    for filename, (url, digest) in FILES.items():
        path = folder / filename
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=90).read())
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Natural Earth input checksum differs: ' + filename)
        archive = zipfile.ZipFile(io.BytesIO(data))
        versions=[n for n in archive.namelist() if n.endswith('VERSION.txt')]
        sources.append({'file': filename, 'url': url, 'sha256': digest, 'bytes': len(data), 'version':archive.read(versions[0]).decode().strip() if versions else 'pinned by hash'})
        stem = {'natural-earth.zip':'ne_10m_admin_0_countries','ne-map-units.zip':'ne_10m_admin_0_map_units','ne-admin1.zip':'ne_10m_admin_1_states_provinces'}[filename]
        reader = shapefile.Reader(shp=io.BytesIO(archive.read(stem+'.shp')), dbf=io.BytesIO(archive.read(stem+'.dbf')), shx=io.BytesIO(archive.read(stem+'.shx')), encoding='utf8')
        if filename in ('natural-earth.zip','ne-map-units.zip'):
            for item in reader.iterShapeRecords():
                row = item.record.as_dict(); name = row['GEOUNIT'] if filename=='ne-map-units.zip' and row['GEOUNIT'] in territories else row['ADMIN']
                for key in ['ADMIN', 'NAME', 'NAME_LONG', 'BRK_NAME', 'FORMAL_EN', 'NAME_EN']:
                    value=row.get(key)
                    if filename=='natural-earth.zip':
                        add(value,name)
                        if value and value!='-99':country_aliases[value.strip().casefold()]=name
                if filename=='natural-earth.zip':continue
                if row['GEOUNIT'] in territories:
                    for key in ['GEOUNIT','NAME','NAME_LONG','NAME_EN']:
                        value=row.get(key)
                        if value and value!='-99':country_aliases[value.strip().casefold()]=name
                shape = item.shape; offsets = list(shape.parts) + [len(shape.points)]
                rings = [shape.points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
                geometry.append({'name': name, 'rings': rings})
        else:
            for record in reader.iterRecords():
                row = record.as_dict(); name = row['admin']
                for key in ['name', 'name_en', 'name_local']:
                    add(row.get(key), name)
    for alias, name in {'UK':'United Kingdom','United States':'United States of America','USA':'United States of America','US':'United States of America','U.S.':'United States of America','Russian Federation':'Russia','Korea, Republic of':'South Korea','Czech Republic':'Czechia','Sao Paulo':'Brazil','Quebec':'Canada','Newfoundland':'Canada','England':'United Kingdom','Scotland':'United Kingdom','Wales':'United Kingdom','Northern Ireland':'United Kingdom'}.items():
        add(alias, name)
        country_aliases[alias.casefold()]=name
    for alias,name in {'Curacao':'Curaçao','Swaziland':'eSwatini','Macedonia':'North Macedonia','Congo (Kinshasa)':'Democratic Republic of the Congo',"Cote D'Ivoire":'Ivory Coast'}.items():
        country_aliases[alias.casefold()]=name
    exact = {k: next(iter(v)) for k, v in aliases.items() if len(v) == 1}
    for name in territories:
        exact[name.casefold()]=name
    ambiguous = {k: sorted(v) for k, v in aliases.items() if len(v) > 1}
    result = {'version':1,'sources':sources,'assignment_unit':'Natural Earth ADMIN except separately mapped overseas GEOUNIT territories; UK home nations combined as United Kingdom; not sovereignty or nationality','coordinate_order':'stored [latitude, longitude], passed to polygon [longitude, latitude]','boundary_policy':'unassigned coastline/border points remain unresolved by coordinates; a unique exact state alias can still assign if no contradictory coordinate country; multi-country hits unresolved','aliases':exact,'country_aliases':country_aliases,'ambiguous_aliases':ambiguous}
    (ROOT/'evidence/brain/geographic-crosswalk.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    payload = {'geography': geometry, 'aliases': exact, 'country_aliases':country_aliases, 'ambiguous_aliases': ambiguous}
    return base64.b64encode(zlib.compress(json.dumps(payload,separators=(',',':')).encode(),9)).decode(), result
