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
BINARY = Path(os.environ['FORGE_BINARY_DIR']) / 'forge_solve'


class BoundSnapshotTests(unittest.TestCase):
    def run_solver(self, directory, *, limited=False, missing=False):
        base = Path(directory) / 'snapshot-é-"quote\\slash\tcontrol'
        env = os.environ.copy()
        for key in ('FORGE_DUMP_MIP_BOUNDS', 'FORGE_DUMP_ROOT_BOUNDS0', 'FORGE_DUMP_ROOT_BOUNDS'):
            env.pop(key, None)
        env['FORGE_DUMP_MIP_BOUNDS'] = str(base)
        if missing:
            env['FORGE_DUMP_ROOT_BOUNDS0'] = str(Path(directory) / 'absent' / 'bounds')
        def limit_files():
            import resource
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
        # Exercise snapshot I/O in one tree. Conditional decomposition owns
        # separate child proof budgets and can legitimately return Feasible
        # under load even after visiting all boundary contexts. That routing
        # is covered by the component/oracle suites; it must not determine
        # whether escaped paths and filesystem errors are tested here.
        # n5-3 only needs to reach the root snapshot phases; its short budget
        # keeps the file-size-limited run cheap and either status is valid.
        model = 'n5-3' if limited else 'p0033'
        budget = '0.1' if limited else '30'
        model_path = ROOT / 'benchmarks/miplib-easy/mps' / (model + '.mps')
        if not model_path.is_file():
            self.skipTest(f'instance data not present: {model_path}')
        command = [str(BINARY), str(model_path), '--engine', 'milp',
                   '--time-limit', budget, '--bab-threads', '1',
                   '--no-component-solve']
        if limited:
            # Exercise an oversized write directly. A decomposed or presolved
            # child can have a small received box that fits under the limit.
            command += ['--no-milp-presolve']
        result = subprocess.run(
            command,
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=limit_files if limited else None, timeout=60)
        # The large model may exhaust the short solve budget with an
        # incumbent (CLI exit 6) or without one (exit 4); this test concerns
        # diagnostic I/O only.
        self.assertIn(result.returncode, (0, 4, 6) if limited else (0,),
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
