from telecom_rag.request_context import conversation_question, sanitize_observation


def test_sanitize_observation_keeps_supported_measurements() -> None:
    observation = sanitize_observation(
        {
            "nr_rsrp_dbm": "-95",
            "nr_sinr_db": 10,
            "observation_id": "obs_1",
            "orientation": "north",
            "ignored": "drop-me",
        }
    )

    assert observation == {
        "nr_rsrp_dbm": -95.0,
        "nr_sinr_db": 10.0,
        "observation_id": "obs_1",
        "orientation": "north",
        "observation_source": "user-entered",
    }


def test_sanitize_observation_requires_measurement() -> None:
    assert sanitize_observation({"observation_id": "obs_1"}) is None
    assert sanitize_observation({"nr_rsrp_dbm": "not-a-number"}) is None


def test_conversation_question_bounds_history() -> None:
    history = [
        {"role": "user", "content": f"question-{index}"}
        for index in range(10)
    ]

    output = conversation_question("current", history, max_messages=2)

    assert "question-7" not in output
    assert "question-8" in output
    assert "question-9" in output
    assert output.endswith("Current question:\ncurrent")


def test_conversation_question_returns_plain_question_without_history() -> None:
    assert conversation_question("current", []) == "current"
