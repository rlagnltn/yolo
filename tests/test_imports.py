def test_core_imports():
    import src.detection
    import src.utils.io_utils
    import src.utils.video_utils
    import src.utils.visualization

    assert src.detection.YOLODetector is not None
