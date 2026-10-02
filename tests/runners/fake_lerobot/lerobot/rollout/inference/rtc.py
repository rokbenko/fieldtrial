from typing import Any


def supports_rtc_inference(policy: Any) -> bool:
    return bool(policy.supports_rtc())
