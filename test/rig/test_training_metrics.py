"""One metric line, all the way from a policy to a curve.

The point of this file is the ROUND TRIP, not any one function. A metric has to
cross three hops -- the grep that runs on the far machine, the parser, and the
series the page builds its panels from -- and each of them used to be a closed
list. If any one of them is narrowed again, a policy that emits a new metric
gets no curve and nothing anywhere says so: the run trains, the log has the
number in it, and the panel simply is not there. So the hops are tested
together, against the real ``KEEP_PATTERN``, rather than one at a time.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_training_metrics
"""

import re
import unittest

from actoris_harena.training import metrics
from actoris_harena.training.metrics import MetricError
from actoris_harena.training.progress import build_series, parse_eval_losses, parse_log
from actoris_harena.training.runs import (
    KEEP_PATTERN,
    parse_predictions,
    parse_val_results,
)

HEAD = (
    "INFO 2026-09-05 10:00:00 ot_train.py:500 {'batch_size': 8,\n"
    " 'log_freq': 100,\n"
    " 'steps': 300,\n"
)
TRAIN = (
    "INFO 2026-09-05 10:0{n}:00 ot_train.py:596 "
    "step:{s} loss:{loss} grdn:2.0 mem_gb:7.0\n"
)


def kept(text: str) -> "list[str]":
    """The lines that survive the grep run on the machine holding the log."""
    return [line for line in text.splitlines() if re.search(KEEP_PATTERN, line)]


class TestTheLineFormat(unittest.TestCase):
    def test_a_line_round_trips(self):
        line = metrics.format_line(12000, {"eval/loss": 0.0421})
        self.assertEqual(line, "harena-metric step=12000 eval/loss=0.0421")
        self.assertEqual(
            metrics.parse_line(line), {"step": 12000, "values": {"eval/loss": 0.0421}}
        )

    def test_several_values_share_one_line(self):
        line = metrics.format_line(10, {"eval/psnr": 31.2, "pred/ssim": 0.88})
        parsed = metrics.parse_line(line)
        self.assertEqual(parsed["values"], {"eval/psnr": 31.2, "pred/ssim": 0.88})

    def test_a_name_with_no_namespace_is_refused(self):
        # It would have no panel block to be drawn in, and picking one for it
        # would put a prediction metric under the training loss by accident.
        with self.assertRaises(MetricError):
            metrics.format_line(1, {"psnr": 30.0})

    def test_an_unknown_namespace_is_refused(self):
        with self.assertRaises(MetricError):
            metrics.format_line(1, {"telemetry/psnr": 30.0})

    def test_a_space_in_a_name_is_refused(self):
        # A space would split one metric into two, both of them nonsense.
        with self.assertRaises(MetricError):
            metrics.format_line(1, {"eval/two words": 1.0})

    def test_a_diverged_run_fails_rather_than_plotting(self):
        # NaN cannot be drawn, and writing it down would make a diverged run
        # look like a parse failure rather than a diverged run.
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(MetricError):
                metrics.format_line(1, {"train/loss": bad})

    def test_an_empty_line_is_not_one(self):
        with self.assertRaises(MetricError):
            metrics.format_line(1, {})

    def test_a_line_that_is_not_a_metric_reads_as_none(self):
        self.assertIsNone(metrics.parse_line("INFO ... step:100 loss:1.0"))

    def test_emit_prints_and_returns_the_same_line(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            line = metrics.emit(5, **{"train/loss": 1.5})
        self.assertEqual(buffer.getvalue().strip(), line)


class TestTheWholeTrip(unittest.TestCase):
    """Emitted -> kept by the far-side grep -> parsed -> a series."""

    def log(self) -> str:
        return (
            HEAD
            + TRAIN.format(n=1, s=100, loss="1.5")
            + metrics.format_line(100, {"eval/psnr_central": 31.2})
            + "\n"
            + "INFO 2026-09-05 10:02:00 ot_train.py:627 step 100: eval_loss=0.4210\n"
            + TRAIN.format(n=3, s=200, loss="0.9")
            + "INFO 2026-09-05 10:04:00 ot_train.py:627 step 200: eval_loss=0.3900\n"
        )

    def test_the_far_side_grep_keeps_both_new_kinds(self):
        lines = kept(self.log())
        self.assertTrue(any(metrics.SENTINEL in line for line in lines))
        self.assertTrue(any("eval_loss=" in line for line in lines))

    def test_a_metric_that_did_not_survive_the_grep_is_the_failure_mode(self):
        # Stated as a test because it is invisible everywhere else: the grep
        # runs on the far machine, so a line it drops never leaves the box and
        # no downstream check can notice its absence.
        self.assertEqual(kept("so101-telemetry step=1 a/b=2"), [])

    def test_every_curve_appears_once_the_log_is_parsed(self):
        parsed = parse_log(self.log())
        self.assertIn("eval/loss", parsed["metrics"])
        self.assertIn("eval/psnr_central", parsed["metrics"])
        self.assertIn("train/loss", parsed["metrics"])

    def test_a_metric_the_parser_never_heard_of_still_becomes_a_series(self):
        # The whole point: no edit to progress.py, runs.py or the page.
        log = (
            HEAD
            + TRAIN.format(n=1, s=100, loss="1.5")
            + metrics.format_line(100, {"pred/hausdorff_mm": 4.25})
            + "\n"
        )
        series = parse_log(log)["series"]
        self.assertEqual(series["pred/hausdorff_mm"][0]["value"], 4.25)

    def test_eval_loss_carries_its_exact_step(self):
        # `step:` is rounded by format_big_number above a thousand; this line
        # is an f-string of the integer, so nothing has to reconstruct it.
        text = "INFO 2026-09-05 10:04:00 x.py:627 step 10600: eval_loss=0.39\n"
        (record,) = parse_eval_losses(text)
        self.assertEqual(record["step"], 10600)

    def test_a_metric_line_borrows_the_time_of_the_line_before_it(self):
        # It is a bare print with no timestamp of its own, and without a time
        # it cannot be drawn against wall-clock at all.
        series = parse_log(self.log())["series"]
        self.assertEqual(
            series["eval/psnr_central"][0]["time"], series["train/loss"][0]["time"]
        )

    def test_the_series_are_sorted_by_step(self):
        out_of_order = (
            HEAD
            + TRAIN.format(n=1, s=100, loss="1.5")
            + TRAIN.format(n=2, s=200, loss="0.9")
            + metrics.format_line(200, {"eval/psnr": 2.0})
            + "\n"
            + metrics.format_line(100, {"eval/psnr": 1.0})
            + "\n"
        )
        steps = [p["step"] for p in parse_log(out_of_order)["series"]["eval/psnr"]]
        self.assertEqual(steps, sorted(steps))

    def test_a_run_with_no_extra_metrics_has_only_the_train_ones(self):
        parsed = parse_log(HEAD + TRAIN.format(n=1, s=100, loss="1.5"))
        self.assertTrue(all(m.startswith("train/") for m in parsed["metrics"]))


class TestBuildSeries(unittest.TestCase):
    def test_a_point_with_no_step_is_dropped(self):
        # resolve_steps gives every point one; a point without is a bug
        # upstream, and plotting it at x=0 would be worse than losing it.
        series = build_series([{"loss": 1.0}], [])
        self.assertEqual(series, {})

    def test_labels_are_not_series(self):
        # step_label and friends are rounded, so they are cross-checks.
        series = build_series([{"step": 1, "loss": 1.0, "step_label": 1000.0}], [])
        self.assertEqual(sorted(series), ["train/loss"])


class TestSimRolloutResults(unittest.TestCase):
    """Per-checkpoint success, which only a simulator can produce."""

    VAL = (
        'VAL /c/val/step_5000.json\n{"success_rate": 0.4, "place_err_mm_mean": 12.5}\n'
        'VAL /c/val/step_10000.json\n{"success_rate": 0.8, "place_err_mm_mean": 6.1}\n'
        'VAL /c/selected.json\n{"step": 10000, "val_success": 0.8}\n'
    )

    def test_each_checkpoint_becomes_a_point(self):
        parsed = parse_val_results(self.VAL)
        self.assertEqual(
            [(p["step"], p["value"]) for p in parsed["series"]["eval/success_rate"]],
            [(5000, 0.4), (10000, 0.8)],
        )

    def test_the_step_comes_from_the_file_name_not_the_order(self):
        # A shell glob sorts step_10000 before step_5000, so reading the order
        # would put the curve back to front.
        backwards = "\n".join(reversed(self.VAL.strip().splitlines())) + "\n"
        parsed = parse_val_results(backwards)
        steps = [p["step"] for p in parsed["series"].get("eval/success_rate", [])]
        self.assertEqual(steps, sorted(steps))

    def test_the_selected_checkpoint_is_kept_apart_from_the_curve(self):
        parsed = parse_val_results(self.VAL)
        self.assertEqual(parsed["selected"]["step"], 10000)
        self.assertEqual(len(parsed["series"]["eval/success_rate"]), 2)

    def test_a_real_run_has_no_such_section_and_gets_no_curve(self):
        # The real rig's equivalent is a person judging trials afterwards. It
        # is a different measurement at a different time and must never be
        # drawn on this axis as though the training loop had produced it.
        self.assertEqual(parse_val_results(""), {"series": {}, "selected": None})

    def test_a_truncated_json_is_skipped_rather_than_raising(self):
        parsed = parse_val_results('VAL /c/val/step_1.json\n{"success_rate": 0.4\n')
        self.assertEqual(parsed["series"], {})


class TestTheOldSentinelStillParses(unittest.TestCase):
    """A training log is the only metric record that exists.

    Runs that finished before the pipeline was shared wrote `so101-metric`, and
    those logs are still the evidence for what they did. Nothing emits the old
    name any more, but it is read for ever.
    """

    def test_a_legacy_line_yields_the_same_points(self):
        old = metrics.parse("so101-metric step=12000 eval/loss=0.0421")
        new = metrics.parse("harena-metric step=12000 eval/loss=0.0421")
        self.assertEqual(old, new)
        self.assertEqual(old[0]["values"], {"eval/loss": 0.0421})

    def test_nothing_emits_the_old_name(self):
        self.assertNotIn("so101", metrics.format_line(1, {"train/loss": 1.0}))


if __name__ == "__main__":
    unittest.main()


class TestWorldModelPredictions(unittest.TestCase):
    """prediction.json, reduced to the verdict a reader needs.

    Built so the answer is known by construction: `central` beats holding on
    two of three horizon steps and `touch_a` on none of them.
    """

    PRED = (
        "PRED /o/run/train/hz/checkpoints/010000/prediction/prediction.json\n"
        '{"policy": "harena_dreamzero", "frames": 40, "episodes": [58, 59],\n'
        ' "per_camera": {\n'
        '   "central": {"psnr": [30.0, 28.0, 26.0],\n'
        '               "psnr_baseline": [29.0, 27.0, 27.0],\n'
        '               "ssim": [0.9, 0.8, 0.7],\n'
        '               "ssim_baseline": [0.8, 0.7, 0.75]},\n'
        '   "touch_a": {"psnr": [40.0, 39.0, 38.0],\n'
        '               "psnr_baseline": [44.0, 43.0, 42.0],\n'
        '               "ssim": [0.99, 0.98, 0.97],\n'
        '               "ssim_baseline": [0.995, 0.99, 0.985]}}}\n'
    )

    def test_the_verdict_is_how_many_steps_it_beats_holding(self):
        [entry] = parse_predictions(self.PRED)
        self.assertEqual(entry["cameras"]["central"]["beats"], 2)
        self.assertEqual(entry["cameras"]["central"]["horizon"], 3)

    def test_a_high_psnr_that_loses_to_holding_is_not_a_win(self):
        # touch_a scores 40 dB and central 30, and touch_a is the FAILURE: a
        # gel image barely moves until contact, so holding the last frame
        # scores higher still. Ranking the cameras by PSNR would report the
        # camera the model has learned nothing about as its best result.
        [entry] = parse_predictions(self.PRED)
        self.assertEqual(entry["cameras"]["touch_a"]["beats"], 0)
        self.assertGreater(
            entry["cameras"]["touch_a"]["psnr"][0],
            entry["cameras"]["central"]["psnr"][0],
        )

    def test_the_step_comes_from_the_path(self):
        # prediction.json does not record which checkpoint it scored; the
        # directory it was written beside is the only place the number is.
        [entry] = parse_predictions(self.PRED)
        self.assertEqual(entry["step"], 10000)

    def test_a_prediction_written_elsewhere_has_no_step_rather_than_a_guessed_one(self):
        moved = self.PRED.replace(
            "/o/run/train/hz/checkpoints/010000/prediction", "/tmp/somewhere"
        )
        [entry] = parse_predictions(moved)
        self.assertIsNone(entry["step"])

    def test_several_checkpoints_read_forwards_whatever_the_glob_gave(self):
        # A shell glob sorts checkpoints/010000 before checkpoints/002000.
        early = self.PRED.replace("/010000/", "/002000/")
        parsed = parse_predictions(self.PRED + early)
        self.assertEqual([e["step"] for e in parsed], [2000, 10000])

    def test_a_policy_that_predicts_no_future_has_no_section_and_no_entry(self):
        # Which is most of them: ACT, diffusion and pi0.5 predict actions only.
        self.assertEqual(parse_predictions(""), [])

    def test_a_truncated_json_is_skipped_rather_than_raising(self):
        self.assertEqual(
            parse_predictions('PRED /a/prediction.json\n{"frames": 2\n'), []
        )

    def test_a_file_with_no_cameras_is_not_an_empty_result(self):
        # An empty per_camera means the scoring run produced nothing -- every
        # frame skipped, say. Reporting it as a scored checkpoint with no
        # cameras would read as "it was scored and had nothing to say".
        self.assertEqual(
            parse_predictions('PRED /a/prediction.json\n{"per_camera": {}}\n'), []
        )
