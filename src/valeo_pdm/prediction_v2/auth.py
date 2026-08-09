from __future__ import annotations

import hmac


def valid_bearer_token(authorization: str | None, expected_token: str | None) -> bool:
    if not expected_token or not expected_token.strip() or not authorization:
        return False
    scheme, separator, token = authorization.partition(" ")
    return bool(
        separator and scheme == "Bearer" and token and hmac.compare_digest(token, expected_token)
    )
