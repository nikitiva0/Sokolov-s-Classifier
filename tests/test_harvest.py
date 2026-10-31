"""Offline tests for tools/harvest: crop geometry, idempotent manifest, threshold labels, HLS parsing."""
from __future__ import annotations
import json
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from harvest import COLUMNS, Layout  # noqa: E402
from harvest.geometry import crop_box, crop_face, timestamps  # noqa: E402
from harvest.labeling import classify, identity, load_thresholds  # noqa: E402
from harvest.manifest import append_rows, next_index, read_rows, refresh_sources, update_rows  # noqa: E402

T = {'solo_min': .3, 'pos': .4, 'neg': .2, 'flag': .2, 'near_duplicate': .97, 'v1_cos': .92, 'v1_phash': 6,
     'v1_source_fraction': .25}


def row(i, **changes):
    base = {'image_id': f'wp2_src_{i:04d}', 'path': f'data/raw/web_positive_v2/faces/src/wp2_src_{i:04d}.jpg',
            'label': 1, 'source_id': 'src', 'group_id': 'src', 'domain': 'talk_stage', 'person': 'sokolov',
            'source_url': 'https://example.org/v', 'timestamp_seconds': f'{10 * i:.3f}', 'det_score': '0.9000',
            'face_px': '64', 'sokolov_similarity': '0.5000', 'review_status': 'accepted', 'notes': ''}
    return base | changes


class CropGeometryTests(unittest.TestCase):
    def test_square_two_times_face_centered(self):
        self.assertEqual(crop_box([100, 100, 150, 160], 640, 480), (65, 70, 185, 190))

    def test_shifted_inside_frame_not_shrunk(self):
        self.assertEqual(crop_box([0, 0, 40, 50], 640, 480), (0, 0, 100, 100))
        self.assertEqual(crop_box([600, 440, 630, 470], 640, 480), (580, 420, 640, 480))

    def test_shrunk_only_when_frame_smaller(self):
        self.assertEqual(crop_box([10, 10, 90, 110], 120, 100), (0, 0, 100, 100))
        left, top, right, bottom = crop_box([-5, 20, 40, 60], 240, 116)
        self.assertEqual((right - left, bottom - top), (90, 90))
        self.assertTrue(0 <= left and right <= 240 and 0 <= top and bottom <= 116)

    def test_never_upscale_downscale_above_512(self):
        from PIL import Image
        crop, box = crop_face(Image.new('RGB', (1920, 1080)), [700, 200, 1100, 700])
        self.assertEqual((box[2] - box[0], crop.size), (1000, (512, 512)))
        crop, box = crop_face(Image.new('RGB', (240, 116)), [100, 30, 140, 80])
        self.assertEqual(crop.size, (100, 100))

    def test_timestamps_even_and_deterministic(self):
        self.assertEqual(timestamps(100, count=4), [12.5, 37.5, 62.5, 87.5])
        self.assertEqual(timestamps(100, 20, 50, every=10), [25.0, 35.0, 45.0])
        self.assertEqual(timestamps(100, end=5, every=10), [2.5])
        with self.assertRaises(ValueError):
            timestamps(100, count=3, every=5)
        with self.assertRaises(ValueError):
            timestamps(100, 60, 50, count=3)


class ManifestTests(unittest.TestCase):
    def test_append_is_idempotent_and_ordered(self):
        with TemporaryDirectory() as folder:
            layout = Layout('web_positive_v2', Path(folder))
            self.assertEqual(len(append_rows(layout.manifest, layout.lock, [row(1), row(2)])), 2)
            self.assertEqual(append_rows(layout.manifest, layout.lock, [row(1), row(2)]), [])
            # Same face under a new id (re-run of the same frame) is not duplicated either.
            self.assertEqual(append_rows(layout.manifest, layout.lock, [row(9) | {'timestamp_seconds': '10.000'}]), [])
            with self.assertRaisesRegex(ValueError, 'занят'):
                append_rows(layout.manifest, layout.lock, [row(1, det_score='0.7000')])
            self.assertEqual(len(append_rows(layout.manifest, layout.lock, [row(3)])), 1)
            rows = read_rows(layout.manifest)
            self.assertEqual([r['image_id'] for r in rows], ['wp2_src_0001', 'wp2_src_0002', 'wp2_src_0003'])
            self.assertEqual(layout.manifest.read_text().splitlines()[0], ','.join(COLUMNS))
            self.assertEqual(next_index(rows, 'wp2_src_'), 4)

    def test_validation_status_update_and_sources(self):
        with TemporaryDirectory() as folder:
            layout = Layout('web_positive_v2', Path(folder))
            for bad in (row(1, label=2), row(1, review_status='ok'), row(1, domain='tv')):
                with self.assertRaises(ValueError):
                    append_rows(layout.manifest, layout.lock, [bad])
            append_rows(layout.manifest, layout.lock, [row(1), row(2, label=0, person='unknown'), row(3)])
            refresh_sources(layout, {'src': {'url': 'u', 'frames_sampled': 5, 'bytes_downloaded': 100}})
            refresh_sources(layout, {'src': {'frames_sampled': 2, 'bytes_downloaded': 50}})

            def exclude(rows):
                rows[2]['review_status'] = 'excluded'
                return 1
            update_rows(layout.manifest, layout.lock, exclude)
            source, = refresh_sources(layout)
            self.assertEqual((source['frames_sampled'], source['bytes_downloaded']), (7, 150))
            self.assertEqual((source['faces_accepted_label1'], source['faces_accepted_label0']), (1, 1))
            self.assertEqual(source['persons'], ['sokolov', 'unknown'])
            self.assertEqual(json.loads(layout.sources.read_text())[0]['url'], 'u')


class LabelTests(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(classify('solo', .35, T)[:2], (1, 'accepted'))
        self.assertEqual(classify('solo', .25, T)[:2], (1, 'ambiguous'))
        self.assertEqual(classify('multi', .40, T)[:2], (1, 'accepted'))
        self.assertEqual(classify('multi', .20, T)[:2], (0, 'accepted'))
        self.assertEqual(classify('multi', .35, T)[:2], (1, 'ambiguous'))
        self.assertEqual(classify('multi', .25, T)[:2], (0, 'ambiguous'))
        self.assertEqual(classify('negative', .19, T)[:2], (0, 'accepted'))
        self.assertEqual(classify('negative', .20, T)[:2], (0, 'ambiguous'))
        with self.assertRaises(ValueError):
            classify('other', .5, T)

    def test_identity_groups(self):
        self.assertEqual(identity(1, 'lec'), ('sokolov', 'lec'))
        self.assertEqual(identity(0, 'lec'), ('unknown', 'lec'))
        self.assertEqual(identity(0, 'lec', 'ivanov'), ('ivanov', 'person:ivanov'))
        self.assertEqual(identity(0, 'lec', 'ivanov', primary=False), ('unknown', 'lec'))

    def test_threshold_file_checked(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 't.json'
            path.write_text(json.dumps({'thresholds': T}))
            self.assertEqual(load_thresholds(path), T)
            path.write_text(json.dumps({'thresholds': T | {'neg': .5}}))
            with self.assertRaises(ValueError):
                load_thresholds(path)


class FakeEngine:
    """Faces with fixed embeddings; FIXTURE ONLY."""
    def __init__(self, vectors):
        self.vectors = vectors

    def detect(self, image):
        return [{'bbox': [20 + 60 * i, 20, 60 + 60 * i, 70], 'confidence': c, 'face_px': px,
                 'landmarks': [[0, 0]] * 5, 'index': i} for i, (c, px, _) in enumerate(self.vectors)]

    def describe(self, image, face):
        return self.vectors[face['index']][2], 12345


class SessionTests(unittest.TestCase):
    def test_labels_crops_near_duplicates_and_rerun(self):
        import numpy as np
        from PIL import Image
        from harvest.commands import Session
        centroid = np.eye(512, dtype=np.float32)[0]
        sokolov = np.eye(512, dtype=np.float32)[0]
        other = np.eye(512, dtype=np.float32)[1]
        args = Namespace(min_det=.6, min_face=28, max_faces=0, mode='multi', person=None, domain='talk_stage')
        engine = FakeEngine([(.9, 50, sokolov), (.9, 40, other), (.5, 40, other), (.9, 20, other)])
        with TemporaryDirectory() as folder:
            layout = Layout('web_positive_v2', Path(folder))
            session = Session(layout, 'talk1', args, T, {'centroid': centroid}, engine)
            rows = session.add(Image.new('RGB', (320, 120)), 12.5, 'https://example.org/v')
            self.assertEqual([(r['label'], r['person'], r['review_status']) for r in rows],
                             [(1, 'sokolov', 'accepted'), (0, 'unknown', 'accepted')])
            self.assertEqual(session.counts['skipped_low_det_or_small'], 2)
            self.assertTrue((Path(folder) / rows[0]['path']).is_file())
            with Image.open(Path(folder) / rows[0]['path']) as crop:
                self.assertEqual(crop.size, (100, 100))
            # The same faces again (another frame) are near-duplicates of saved ones.
            self.assertEqual(session.add(Image.new('RGB', (320, 120)), 15.0, 'https://example.org/v'), [])
            self.assertEqual(session.counts['skipped_near_duplicate'], 2)
            # A fresh session (re-run) keeps numbering and skips the already stored faces.
            again = Session(layout, 'talk1', args, T, {'centroid': centroid}, engine)
            self.assertEqual(again.next, 3)
            self.assertEqual(again.add(Image.new('RGB', (320, 120)), 12.5, 'https://example.org/v'), [])
            self.assertEqual(len(read_rows(layout.manifest)), 2)
            with np.load(layout.embeddings('talk1')) as store:
                self.assertEqual(store['emb'].shape, (2, 512))


class NetworkParsingTests(unittest.TestCase):
    def test_hls_playlist_variant_byterange_and_segment_choice(self):
        from harvest import net
        master = '#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=1920x1080\nhi.m3u8\n' \
                 '#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=1280x720\nmid.m3u8\n'
        media = '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:4.0,\n#EXT-X-BYTERANGE:1000@0\nseg.mp4\n' \
                '#EXTINF:4.0,\n#EXT-X-BYTERANGE:500\nseg.mp4\n#EXTINF:2.5,\nhttps://cdn.example/x.ts\n'
        pages = {'https://h.example/a/master.m3u8': master, 'https://h.example/a/mid.m3u8': media}
        with patch.object(net, 'http_get', lambda url, *a, **k: pages[url].encode()):
            hls = net.HLS('https://h.example/a/master.m3u8', max_height=720)
        self.assertEqual(hls.url, 'https://h.example/a/mid.m3u8')
        self.assertEqual(hls.duration, 10.5)
        self.assertEqual(hls.init, ('https://h.example/a/init.mp4', None))
        self.assertEqual([s[3] for s in hls.segments], [(0, 999), (1000, 1499), None])
        self.assertEqual(hls.segments[2][2], 'https://cdn.example/x.ts')
        encrypted = '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="k"\n#EXTINF:4,\ns.ts\n'
        with patch.object(net, 'http_get', lambda *a, **k: encrypted.encode()):
            with self.assertRaisesRegex(net.FetchError, 'зашифрован'):
                net.HLS('https://h.example/e.m3u8')

    def test_format_choice(self):
        from harvest import net
        info = {'formats': [
            {'format_id': 'a', 'vcodec': 'none', 'url': 'u', 'protocol': 'https'},
            {'format_id': 'av1', 'vcodec': 'av01.0.08M.08', 'height': 720, 'url': 'u', 'protocol': 'https'},
            {'format_id': 'mp4', 'vcodec': 'avc1', 'height': 720, 'tbr': 900, 'url': 'u', 'protocol': 'https'},
            {'format_id': 'hls', 'vcodec': 'avc1', 'height': 720, 'tbr': 950, 'url': 'u', 'protocol': 'm3u8_native'},
            {'format_id': 'big', 'vcodec': 'avc1', 'height': 1080, 'url': 'u', 'protocol': 'https'},
            {'format_id': 'dash', 'vcodec': 'avc1', 'height': 720, 'url': 'u', 'protocol': 'http_dash_segments'}]}
        self.assertEqual(net.pick_format(info, 720)['format_id'], 'hls')
        self.assertEqual(net.pick_format(info, 1080)['format_id'], 'big')
        self.assertEqual(net.pick_format(info, 720, max_kbps=920)['format_id'], 'mp4')
        with self.assertRaises(net.FetchError):
            net.pick_format(info, 480)

    def test_hamming(self):
        from harvest.faces import hamming
        self.assertEqual(hamming([0, 0xFF], [0, 2 ** 64 - 1]).tolist(), [[0, 64], [8, 56]])


if __name__ == '__main__':
    unittest.main()
