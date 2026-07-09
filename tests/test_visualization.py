from src.utils.visualization import draw_detections


class DummyFrame:
    def __init__(self):
        self.copied = False

    def copy(self):
        copied = DummyFrame()
        copied.copied = True
        return copied


def test_draw_detections_handles_empty_object_list_without_cv2():
    frame = DummyFrame()

    annotated = draw_detections(frame, [])

    assert annotated.copied is True
