"""Build the FlyWire v783 wiring used by the simulations: data/v783/fafb783-connectome.npz.

Usage, from the repository root:
  python scripts/build_v783.py

Two pinned sources, both listed in data/sources.json with their licences:
  * the Bates/Lee lab compiled FlyWire v783 metadata (cell type, super class), downloaded whole (13.5 MB);
  * the v783 connectivity table distributed with Shiu et al. 2024. Only four columns are needed, so the script fetches
    the Parquet footer and those columns' chunks with HTTP range requests (47.4 MB of the 100.8 MB file).
Each download is checked against its recorded SHA-256 (for the connectivity table, the hash covers exactly the byte
ranges fetched, in file order). Downloads are kept in data/downloads/ and reused on later runs.

The archive holds, in FlyWire root-ID order, each cell's ID, cell-type label and super class (most cells are neurons;
about 6,800 carry only coarse labels and are mostly putative glia), and for every
connection whose two neurons both have metadata: presynaptic and postsynaptic index, synapse count and Shiu et al.'s
excitatory (+1) or inhibitory (-1) sign. A neuron without a cell type is labelled by its sub-class, class or super
class followed by "(untyped)". Needs pyarrow.
"""
import hashlib, io, json, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / 'data/sources.json').read_text())['v783']
CACHE = ROOT / 'data/downloads'
OUT = ROOT / SOURCES['output']['file']
FOOTER_WINDOW = 65536


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, start=None, end=None):
    """GET a URL, or the byte range [start, end), and return the body."""
    headers = {'User-Agent': 'connectome-null-wiring-benchmark'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end - 1}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
        data = response.read()
    if start is not None and len(data) != end - start:
        raise IOError(f'Expected {end - start} bytes from {url}, received {len(data)}.')
    return data


def metadata_file():
    source = SOURCES['metadata']
    path = ROOT / source['file']
    if not path.is_file():
        print(f'download {source["url"]} ({source["bytes"] / 1e6:.1f} MB)', flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.part')
        part.write_bytes(fetch(source['url']))
        part.rename(path)
    if sha256(path.read_bytes()) != source['sha256']:
        sys.exit(f'{path.name} does not match the SHA-256 in data/sources.json. Delete it to download again.')
    return path


class PartialFile(io.RawIOBase):
    """A read-only view of a remote file that serves only the byte ranges already downloaded."""

    def __init__(self, size, ranges):
        self.size, self.ranges, self.pos = size, sorted(ranges.items()), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        for start, data in self.ranges:
            if start <= self.pos and self.pos + n <= start + len(data):
                buffer[:n] = data[self.pos - start:self.pos - start + n]
                self.pos += n
                return n
        raise IOError(f'Byte range {self.pos}-{self.pos + n} was not downloaded.')


def download_ranges(source, blob_path, index_path):
    """Fetch the Parquet footer, work out the needed column chunks, and fetch them."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    size = source['upstream_bytes']
    tail_start = size - FOOTER_WINDOW
    print(f'read the Parquet footer of {source["url"]}', flush=True)
    tail = fetch(source['url'], tail_start, size)
    footer = pq.ParquetFile(pa.PythonFile(PartialFile(size, {tail_start: tail}), mode='r')).metadata
    wanted = set(source['columns'])
    spans = []
    for g in range(footer.num_row_groups):
        group = footer.row_group(g)
        for c in range(group.num_columns):
            column = group.column(c)
            if column.path_in_schema in wanted:
                start = column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
                spans.append([start, start + column.total_compressed_size])
    merged = []
    for start, end in sorted(spans):
        if merged and merged[-1][1] == start:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    total = sum(end - start for start, end in merged)
    print(f'fetch {len(merged)} column-chunk ranges, {total / 1e6:.1f} MB of {size / 1e6:.1f} MB', flush=True)
    ranges, chunks = [[tail_start, size]], [tail]
    for start, end in merged:
        if end > tail_start:
            raise IOError('A column chunk overlaps the footer read; enlarge FOOTER_WINDOW.')
        chunks.append(fetch(source['url'], start, end))
        ranges.append([start, end])
    order = np.argsort([r[0] for r in ranges])
    blob = b''.join(chunks[i] for i in order)
    ranges = [ranges[i] for i in order]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    part = blob_path.with_name(blob_path.name + '.part')
    part.write_bytes(blob)
    part.rename(blob_path)
    index_path.write_text(json.dumps({'url': source['url'], 'upstream_bytes': size, 'ranges': ranges,
                                      'sha256': sha256(blob)}, indent=1))


def connectivity_columns():
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = SOURCES['connectivity']
    blob_path = ROOT / source['file']
    index_path = blob_path.with_name(blob_path.name + '.json')
    if not blob_path.is_file() or not index_path.is_file():
        download_ranges(source, blob_path, index_path)
    index = json.loads(index_path.read_text())
    blob = blob_path.read_bytes()
    if sha256(blob) != source['sha256'] or index['sha256'] != source['sha256']:
        sys.exit(f'{blob_path.name} does not match the SHA-256 in data/sources.json. '
                 f'Delete it and {index_path.name} to download again.')
    pieces, offset = {}, 0
    for start, end in index['ranges']:
        pieces[start] = blob[offset:offset + end - start]
        offset += end - start
    parquet = pq.ParquetFile(pa.PythonFile(PartialFile(source['upstream_bytes'], pieces), mode='r'), pre_buffer=False)
    table = parquet.read(columns=source['columns'], use_threads=False)
    return {name: table.column(name).to_numpy() for name in source['columns']}


def text(values):
    return np.array(['' if v is None else str(v) for v in values], dtype=object)


def load_metadata(path):
    import pyarrow.feather as feather
    table = feather.read_table(path)
    meta = {n: text(table.column(n).to_pylist()) for n in ('super_class', 'cell_class', 'cell_sub_class', 'cell_type')}
    meta['id'] = np.array([int(v) for v in table.column('fafb_783_id').to_pylist()], dtype=np.int64)
    if len(np.unique(meta['id'])) != len(meta['id']):
        sys.exit('The metadata contains duplicate neuron IDs.')
    return meta


def encode(values):
    table, index = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return table, index.astype(np.int32)


def main():
    meta = load_metadata(metadata_file())
    columns = connectivity_columns()
    order = np.argsort(meta['id'])
    meta = {key: values[order] for key, values in meta.items()}
    ids = meta['id']

    def index_of(values):
        position = np.searchsorted(ids, values)
        position[position >= len(ids)] = 0
        return np.where(ids[position] == values, position, -1)

    pre = index_of(columns['Presynaptic_ID'].astype(np.int64))
    post = index_of(columns['Postsynaptic_ID'].astype(np.int64))
    matched = (pre >= 0) & (post >= 0)
    labels = np.array([t or f"{sub or cls or sup or 'unannotated'} (untyped)" for t, sub, cls, sup in
                       zip(meta['cell_type'], meta['cell_sub_class'], meta['cell_class'], meta['super_class'])],
                      dtype=object)
    types, type_index = encode(labels)
    supers, super_index = encode(np.where(meta['super_class'] == '', 'unknown', meta['super_class']))
    header = {'neurons': int(len(ids)), 'connections': int(matched.sum()),
              'connections_without_metadata': int((~matched).sum()),
              'sources': {k: {f: SOURCES[k][f] for f in ('url', 'sha256', 'license')} for k in ('metadata', 'connectivity')}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, header=np.array(json.dumps(header)), neuron_id=ids, cell_type=type_index, super_class=super_index,
        pre=pre[matched].astype(np.int32), post=post[matched].astype(np.int32),
        synapses=columns['Connectivity'][matched].astype(np.int32),
        shiu_sign=columns['Excitatory'][matched].astype(np.int8),
        lookup_cell_type=types, lookup_super_class=supers)
    expected = SOURCES['output']
    ok = header['neurons'] == expected['neurons'] and header['connections'] == expected['connections']
    print(f'wrote {OUT.relative_to(ROOT)}: {header["neurons"]:,} cells, {header["connections"]:,} connections '
          f'({header["connections_without_metadata"]:,} without metadata left out), {len(types):,} cell-type labels'
          + ('' if ok else '  WARNING: counts differ from data/sources.json'))


if __name__ == '__main__':
    main()
