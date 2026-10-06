"""A RealSense whose colour stops while its depth runs on is caught, not recorded.

Measured on the UR3e cell's D435 (USB 2.1): a pipeline opened right after
another was stopped delivered colour frame 2 and then nothing, while depth ran
to frame 104. Every frameset paired fresh depth with the same colour, nothing
timed out, and a session recorded 1500 byte-identical pictures. These drive the
capture loop with a fake pipeline: no camera needed, pyrealsense2 imported only
because the module imports it.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import actoris_harena.recording.realsense_camera as rc
except ImportError:  # pragma: no cover - a venv without pyrealsense2
    rc = None  # type: ignore[assignment]


class _Frame:
    def __init__(self, number, value):
        self.number = number
        self.data = np.full((2, 3, 3), value, np.uint8)

    def __bool__(self):
        return True

    def get_frame_number(self):
        return self.number

    def get_data(self):
        return self.data


class _Frames:
    def __init__(self, colour, depth):
        self.colour, self.depth = colour, depth

    def get_color_frame(self):
        return self.colour

    def get_depth_frame(self):
        return self.depth


class _Pipeline:
    """Serves ``colour_numbers`` in order with ever-advancing depth."""

    def __init__(self, colour_numbers, then_stop):
        self.numbers = list(colour_numbers)
        self.then_stop = then_stop
        self.depth = 0
        self.served = []

    def try_wait_for_frames(self, _timeout_ms):
        if not self.numbers:
            self.then_stop()
            return False, None
        n = self.numbers.pop(0)
        self.depth += 1
        depth = _Frame(self.depth, 0)
        depth.data = np.full((2, 3), 500, np.uint16)
        frames = _Frames(_Frame(n, n % 250), depth)
        self.served.append(frames)
        return True, frames

    def stop(self):
        pass


class _Publisher:
    def __init__(self):
        self.rgb = []
        self.depth = 0
        self.stopped = False

    def is_shutdown_requested(self):
        return self.stopped

    def set_rgb_image(self, rgb, name, t_capture=None):
        self.rgb.append(rgb)

    def set_depth_image(self, depth, name, t_capture=None):
        self.depth += 1


@unittest.skipIf(rc is None, "pyrealsense2 is not installed")
class TestAFrozenColourStreamIsCaught(unittest.TestCase):
    def _capture(self, numbers, restarts):
        cap = rc.RealSenseCapture(
            rgb_name="top", depth_name="top_depth", width=3, height=2, fps=30
        )
        cap.lock_auto_exposure = False
        publisher = _Publisher()

        def stop():
            publisher.stopped = True

        cap._pipeline = _Pipeline(numbers, stop)

        def restart():
            # The end of the fake stream is not a stall worth counting.
            if not publisher.stopped:
                restarts.append(True)
            return True

        cap._restart = restart
        return cap, publisher

    def test_a_repeated_colour_frame_is_not_published_with_its_depth(self):
        restarts = []
        cap, publisher = self._capture([1, 2, 2, 2, 3], restarts)
        with mock.patch.object(rc, "_COLOUR_STALL_S", 60.0):
            cap._loop(publisher)
        self.assertEqual(len(publisher.rgb), 3)  # frames 1, 2 and 3
        self.assertEqual(publisher.depth, 3)  # depth stays 1:1 with colour
        self.assertEqual(restarts, [])

    def test_colour_stuck_while_depth_runs_restarts_the_pipeline(self):
        restarts = []
        cap, publisher = self._capture([1, 2] + [2] * 50, restarts)
        real = cap._restart

        def restart_once():
            real()
            publisher.stopped = True  # look no further than the first restart

        cap._restart = restart_once
        with mock.patch.object(rc, "_COLOUR_STALL_S", 0.0):
            cap._loop(publisher)
        self.assertEqual(len(restarts), 1)
        self.assertEqual(len(publisher.rgb), 2)

    def test_it_gives_up_when_each_restart_gives_one_frame_and_sticks(self):
        # What was measured: a frame, then nothing. One fresh frame after a
        # restart is not recovery, or the loop would restart for ever.
        restarts = []
        cap, publisher = self._capture([1] + [1] * 500, restarts)
        with mock.patch.object(rc, "_COLOUR_STALL_S", 0.0), mock.patch(
            "builtins.print"
        ):
            cap._loop(publisher)  # logs the give-up and returns; never hangs
        self.assertEqual(len(restarts), rc._MAX_RESTARTS)

    def test_the_published_colour_is_a_copy(self):
        # A view would keep librealsense's buffer alive in the frame store's
        # history; the colour sensor's frame pool is small.
        restarts = []
        cap, publisher = self._capture([1], restarts)
        source = cap._pipeline
        cap._loop(publisher)
        self.assertEqual(len(publisher.rgb), 1)
        source.served[0].colour.data[:] = 99  # librealsense reuses the buffer
        self.assertNotEqual(int(publisher.rgb[0][0, 0, 0]), 99)


if __name__ == "__main__":
    unittest.main()
