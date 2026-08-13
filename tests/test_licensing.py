from avatar_face.domain.licensing import (
    ModelCandidate,
    ModelComponent,
    PermissiveLicensePolicy,
)


def test_policy_approves_fixed_permissive_candidate() -> None:
    candidate = ModelCandidate(
        identifier="example/model",
        revision="abc123",
        estimated_download_gib=1.0,
        components=(ModelComponent("weights", "Apache-2.0", "https://example.com"),),
        android_fit_as_is=False,
        role="teacher_candidate",
    )

    audit = PermissiveLicensePolicy().audit(candidate)

    assert audit.approved
    assert audit.findings == ()


def test_policy_rejects_restricted_component_and_missing_revision() -> None:
    candidate = ModelCandidate(
        identifier="example/restricted",
        revision=None,
        estimated_download_gib=None,
        components=(
            ModelComponent(
                "weights",
                "Custom",
                "https://example.com",
                has_use_restrictions=True,
            ),
        ),
        android_fit_as_is=False,
        role="license_rejected",
    )

    audit = PermissiveLicensePolicy().audit(candidate)

    assert not audit.approved
    assert len(audit.findings) == 3
