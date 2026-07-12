import academic_chatbot


def test_package_exposes_version() -> None:
    assert academic_chatbot.__version__ == "0.1.0.dev0"
