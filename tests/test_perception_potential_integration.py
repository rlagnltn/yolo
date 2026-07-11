import numpy as np

from src.perception.pipeline import PerceptionPipeline


class Detector:
    model_name = "det"
    def detect_frame(self, frame): return []


class Segmenter:
    model_name = "seg"
    def segment_frame(self, frame, *args, **kwargs): return []


class Depth:
    model_name = "depth"
    def predict(self, frame): return {"depth_map": np.ones(frame.shape[:2], np.float32), "depth_type": "metric", "unit": "meter", "model_name": "depth"}


class Scene:
    model_name = "scene"
    id2label = {0: "road", 1: "car"}
    def predict(self, frame): return np.array([[0, 1], [0, 0]], np.uint8)


def test_potential_requires_mapping():
    result = PerceptionPipeline(Detector(), Segmenter()).process_frame(
        np.zeros((2, 2, 3), np.uint8), 0, potential_output={"enabled": True, "goal": {"row": 0, "col": 0}}
    )
    assert result["potential"] is None
    assert result["errors"][-1] == "potential: mapping must be enabled before potential generation."


def test_potential_metadata_from_mock_mapping(tmp_path):
    result = PerceptionPipeline(Detector(), Segmenter(), scene_segmenter=Scene(), depth_estimator=Depth()).process_frame(
        np.zeros((2, 2, 3), np.uint8), 0,
        scene_output={"class_map_dir": tmp_path, "color_map_dir": tmp_path, "region_dir": tmp_path, "save_class_maps": False, "save_color_maps": False, "save_regions": False},
        depth_output={"raw_depth_dir": tmp_path, "depth_png_dir": tmp_path, "color_map_dir": tmp_path, "visualization_dir": tmp_path, "save_raw_depth": False, "save_depth_png": False, "save_color_maps": False, "save_visualizations": False},
        geometry_output={"enabled": True, "point_cloud_dir": tmp_path, "stride": 1, "min_depth_m": 0.1, "max_depth_m": 80.0, "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2}},
        bev_output={"enabled": True, "config": {"x_min_m": -1, "x_max_m": 2, "z_min_m": 0, "z_max_m": 2, "resolution_m": 1}, "id2label": Scene.id2label, "class_grid_dir": tmp_path, "drivable_grid_dir": tmp_path, "non_drivable_grid_dir": tmp_path, "visualization_dir": tmp_path, "save_class_grid_npy": False, "save_class_grid_png": False, "save_region_masks": False, "save_visualizations": False},
        mapping_output={"enabled": True, "config": {}, "occupancy_dir": tmp_path, "cost_grid_dir": tmp_path, "inflated_cost_dir": tmp_path, "visualization_dir": tmp_path, "save_occupancy_npy": False, "save_occupancy_png": False, "save_cost_npy": False, "save_cost_png": False, "save_inflated_cost": False, "save_visualizations": False},
        potential_output={"enabled": True, "config": {"attractive": {"saturation_distance_m": None}}, "goal": {"row": 0, "col": 1}, "attractive_dir": tmp_path, "repulsive_dir": tmp_path, "combined_dir": tmp_path, "gradient_dir": tmp_path, "visualization_dir": tmp_path, "save_png": False, "save_visualizations": False},
    )
    assert result["potential"]["grid_type"] == "goal_conditioned_potential"
    assert result["potential"]["combined_path"].endswith("frame_000000.npy")
