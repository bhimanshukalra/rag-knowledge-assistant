from langchain_core.documents import Document


def normalize_access_groups(metadata: dict):
    access_groups = metadata.get("access_groups", "")

    if isinstance(access_groups, str):
        return [group for group in access_groups.split(",") if group]

    return access_groups


def user_can_access(document: Document, user_groups: list[str]):
    allowed_groups = document.metadata.get("access_groups", [])

    return bool(set(allowed_groups) & set(user_groups))


def get_access_flag(group: str):
    return f"access_{group}"


def build_access_metadata(access_groups: list[str]):
    metadata = {"access_groups": ",".join(access_groups)}

    for group in access_groups:
        metadata[get_access_flag(group)] = True

    return metadata


def metadata_matches_access_groups(metadata: dict, access_groups: list[str]):
    expected_access_metadata = build_access_metadata(access_groups)

    return all(
        metadata.get(key) == value for key, value in expected_access_metadata.items()
    )


def build_access_filter(user_groups: list[str]):
    access_filters = [{get_access_flag(group): True} for group in user_groups]

    if len(access_filters) == 1:
        return access_filters[0]

    return {"$or": access_filters}
