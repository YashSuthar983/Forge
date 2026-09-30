"""Parse real diagnostic records and exercise filesystem write failures."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ['SOR_BINARY_DIR']) / 'sor_solve'


class BoundSnapshotTests(unittest.TestCase):
    def run_solver(self, directory, *, limited=False, missing=False):
        base = Path(directory) / 'snapshot-é-"quote\\slash\tcontrol'
        env = os.environ.copy()
        for key in ('SOR_DUMP_MIP_BOUNDS', 'SOR_DUMP_ROOT_BOUNDS0', 'SOR_DUMP_ROOT_BOUNDS'):
            env.pop(key, None)
        env['SOR_DUMP_MIP_BOUNDS'] = str(base)
        if missing:
            env['SOR_DUMP_ROOT_BOUNDS0'] = str(Path(directory) / 'absent' / 'bounds')
        def limit_files():
            import resource
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
        model = 'n5-3' if limited else 'p0033'
        result = subprocess.run(
            [str(BINARY), str(ROOT / 'benchmarks/miplib-easy/mps' / (model + '.mps')),
             '--engine', 'milp', '--time-limit', '0.1', '--bab-threads', '1'],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=limit_files if limited else None, timeout=15)
        # The large model may exhaust the short solve budget without an
        # incumbent (CLI exit 4); this test concerns diagnostic I/O only.
        self.assertIn(result.returncode, (0, 4) if limited else (0,),
                      result.stderr.decode())
        return [json.loads(line) for line in Path(str(base) + '.invocations.jsonl').read_text().splitlines()]

    def test_escaped_paths_parse_and_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_solver(directory)
            snapshots = [r for r in records if r['wrote_bounds']]
            self.assertTrue(snapshots)
            for row in snapshots:
                self.assertIn('"quote\\slash\tcontrol', row['snapshot'])
                body = Path(row['snapshot']).read_bytes()
                self.assertEqual(hashlib.sha256(body).hexdigest(), row['snapshot_sha256'])
                self.assertEqual(row['snapshot_error'], '')

    def test_open_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_solver(directory, missing=True)
            row = next(r for r in records if r['phase'] == 'pre_probing_box')
            self.assertFalse(row['wrote_bounds'])
            self.assertEqual(row['snapshot_sha256'], '')
            self.assertEqual(row['snapshot_error'], 'open_failed')

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX file-size limit')
    def test_short_or_buffered_write_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_solver(directory, limited=True)
            row = next(r for r in records if r['phase'] == 'received_box')
            self.assertFalse(row['wrote_bounds'])
            self.assertEqual(row['snapshot_sha256'], '')
            self.assertIn(row['snapshot_error'], ('write_failed', 'close_failed'))


if __name__ == '__main__':
    unittest.main()
