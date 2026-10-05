from src.mlops.tracking import ExperimentTracker


def test_tree_storage_type_is_trusted():
    # skops 0.16+ refuses to save tree models unless this type is trusted.
    # This failed silently on a fresh install once, so keep a test on it.
    assert "sklearn.tree._tree.Tree" in ExperimentTracker.TRUSTED_TYPES
