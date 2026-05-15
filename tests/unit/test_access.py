from medical_extraction.retrieval.access import role_is_admin, role_is_authorized


def test_doctor_role_is_authorized():
    assert role_is_authorized("doctor") is True


def test_staff_roles_are_redacted():
    assert role_is_authorized("receptionist") is False


def test_admin_role_is_separate_from_authorized_access():
    assert role_is_admin("admin") is True
    assert role_is_authorized("admin") is False
