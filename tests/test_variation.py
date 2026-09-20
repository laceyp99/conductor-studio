from datetime import datetime, timezone

from conductor_studio.models import SessionSettings
from conductor_studio.variation import SLOT_IDS, create_manifest


def test_policy_constructs_one_queued_batch_without_assignments():
    value = create_manifest(
        SessionSettings(prompt="Theme", provider="openai", model="model"),
        core_version="0.5.3",
        session_id="20260920-120000_abcd",
        now=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    assert tuple(slot.slot_id for slot in value.slots) == SLOT_IDS
    assert value.batch.core_version == "0.5.3"
    assert value.batch.generation_ids == []
    assert all("parameters" not in slot for slot in value.model_dump()["slots"])
