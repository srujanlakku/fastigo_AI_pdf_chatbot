from src.cloud_bootstrap import apply_cloud_runtime_patches


def test_cloud_bootstrap_is_idempotent():
    apply_cloud_runtime_patches()
    apply_cloud_runtime_patches()
