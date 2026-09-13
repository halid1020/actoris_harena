"""The two protocols that let one recorder serve two robots.

The recorder is 800 lines of pacing, pausing, sampling, assembling and writing,
and exactly three of its questions are about a robot. These are those questions,
and the properties worth pinning are the ones that decide what a dataset looks
like when something goes wrong.
"""

import unittest

import numpy as np

from actoris_harena.recording.frames import (
    FramePublisher,
    FrameStore,
    ObservationBuilder,
    TimedFrameSource,
)


class _Source:
    """A minimal TimedFrameSource: enough methods, nothing else."""

    def get_rgb_image_at(self, name, t_ref):
        return (np.zeros((2, 2, 3), np.uint8), 0.001)

    def get_depth_image_at(self, name, t_ref):
        return (np.zeros((2, 2), np.uint16), 0.001)

    def get_rgb_image_age(self, name):
        return 0.0

    def is_shutdown_requested(self):
        return False

    def request_shutdown(self):
        pass


class _Builder:
    """A minimal ObservationBuilder."""

    def state_and_action(self, t_ref, drifts):
        drifts["joints"] = 0.002
        return np.zeros(7), np.zeros(7), False

    def ee(self, t_ref, drifts):
        return None

    def teleop_active(self):
        return True

    def armed(self):
        return True


class TestTheContractsAreStructural(unittest.TestCase):
    def test_a_plain_object_with_the_methods_is_a_frame_source(self):
        self.assertIsInstance(_Source(), TimedFrameSource)

    def test_a_plain_object_with_the_methods_is_an_observation_builder(self):
        self.assertIsInstance(_Builder(), ObservationBuilder)

    def test_a_missing_method_is_not(self):
        class Partial:
            def get_rgb_image_at(self, name, t_ref):
                return None

        self.assertNotIsInstance(Partial(), TimedFrameSource)

    def test_publishing_and_looking_up_by_time_are_different_contracts(self):
        # A FrameStore publishes; it does NOT answer "what was the frame at
        # t_ref". Conflating them is how a recorded frame becomes a mix of
        # latest values instead of one instant.
        store = FrameStore()
        self.assertIsInstance(store, FramePublisher)
        self.assertNotIsInstance(store, TimedFrameSource)


class TestWhatNoneMeans(unittest.TestCase):
    """Returning None must skip the frame, never write a partial one."""

    def test_a_builder_with_no_state_yet_says_so(self):
        class NotReady(_Builder):
            def state_and_action(self, t_ref, drifts):
                return None

        self.assertIsNone(NotReady().state_and_action(0.0, {}))

    def test_a_builder_records_its_own_drift_alongside_the_cameras(self):
        # The point of passing the dict in: a rig's proprioception drift sits in
        # the same row as its cameras', so one file shows the whole alignment.
        drifts = {}
        _Builder().state_and_action(1.0, drifts)
        self.assertIn("joints", drifts)

    def test_the_fallback_flag_is_part_of_the_answer(self):
        # A stale command means the action fell back to the measured state,
        # which teaches a hold nobody performed. The recorder tallies it, so the
        # builder has to report it rather than hide it.
        _state, _action, fell_back = _Builder().state_and_action(1.0, {})
        self.assertIs(fell_back, False)

    def test_a_rig_that_records_no_ee_returns_none_rather_than_zeros(self):
        # Zeros would be written into the dataset as a real pose at the origin.
        self.assertIsNone(_Builder().ee(1.0, {}))


class TestArmed(unittest.TestCase):
    def test_a_disarmed_rig_is_reported_so_the_episode_can_be_discarded(self):
        class Disarmed(_Builder):
            def armed(self):
                return False

        # An episode recorded past an e-stop or a torque cut has frames whose
        # action is just the measured state. Saving it short would be worse than
        # discarding it.
        self.assertFalse(Disarmed().armed())


if __name__ == "__main__":
    unittest.main()
