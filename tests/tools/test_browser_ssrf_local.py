"""Tests that browser_navigate SSRF checks respect local-backend mode and
the allow_private_urls setting.

Local backends (Camofox, headless Chromium without a cloud provider) skip
SSRF checks entirely — the agent already has full local-network access via
the terminal tool.

Cloud backends (Browserbase, BrowserUse) enforce SSRF by default.  Users
can opt out for cloud mode via ``browser.allow_private_urls: true``.
"""

import json

import pytest

from tools import browser_tool


def _make_browser_result(url="https://example.com"):
    """Return a mock successful browser command result."""
    return {"success": True, "data": {"title": "OK", "url": url}}


# ---------------------------------------------------------------------------
# Pre-navigation SSRF check
# ---------------------------------------------------------------------------


class TestPreNavigationSsrf:
    PRIVATE_URL = "http://127.0.0.1:8080/dashboard"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for pre-navigation tests that pass the SSRF check."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            },
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(),
        )

    # -- Cloud mode: SSRF active -----------------------------------------------

    def test_cloud_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """SSRF protection blocks private URLs in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_cloud_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Private URLs pass in cloud mode when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_cloud_allows_public_url(self, monkeypatch, _common_patches):
        """Public URLs always pass in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    # -- Local mode: SSRF skipped ----------------------------------------------

    def test_local_allows_private_url(self, monkeypatch, _common_patches):
        """Local backends skip SSRF — private URLs are always allowed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_local_allows_public_url(self, monkeypatch, _common_patches):
        """Local backends pass public URLs too (sanity check)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True


# ---------------------------------------------------------------------------
# _is_local_backend() unit tests
# ---------------------------------------------------------------------------


class TestIsLocalBackend:
    def test_camofox_is_local(self, monkeypatch):
        """Camofox mode counts as a local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "anything")

        assert browser_tool._is_local_backend() is True

    def test_no_cloud_provider_is_local(self, monkeypatch):
        """No cloud provider configured → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

        assert browser_tool._is_local_backend() is True

    def test_cloud_provider_is_not_local(self, monkeypatch):
        """Cloud provider configured and not Camofox → NOT local."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "bb")

        assert browser_tool._is_local_backend() is False


# ---------------------------------------------------------------------------
# Post-redirect SSRF check
# ---------------------------------------------------------------------------


class TestPostRedirectSsrf:
    PUBLIC_URL = "https://example.com/redirect"
    PRIVATE_FINAL_URL = "http://192.168.1.1/internal"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for redirect tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            },
        )

    # -- Cloud mode: redirect SSRF active --------------------------------------

    def test_cloud_blocks_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]

    def test_cloud_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in cloud mode with allow_private_urls."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    # -- Local mode: redirect SSRF skipped -------------------------------------

    def test_local_allows_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in local mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    def test_cloud_allows_redirect_to_public(self, monkeypatch, _common_patches):
        """Redirects to public addresses always pass (cloud mode)."""
        final = "https://example.com/final"
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=final),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == final


# ---------------------------------------------------------------------------
# IMDS blocking with hybrid routing (Issue #16234)
# ---------------------------------------------------------------------------


class TestImdsBlockingWithHybridRouting:
    """Verify IMDS endpoints are blocked even when hybrid routing is enabled.

    This tests the fix for Issue #16234: the pre-navigation SSRF guard must
    run BEFORE the hybrid routing decision, not after. Previously, when
    auto_local_for_private_urls was enabled and a cloud provider was configured,
    the SSRF check was skipped entirely, allowing access to 169.254.169.254.
    """

    AWS_IMDS = "http://169.254.169.254/latest/meta-data/"
    AWS_IMDS_V2 = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
    GCP_IMDS = "http://metadata.google.internal/computeMetadata/v1/instance/hostname"
    AZURE_IMDS = "http://169.254.169.253/metadata/instance"
    ALIYUN_IMDS = "http://100.100.100.200/latest/meta-data/"

    @pytest.fixture()
    def _cloud_mode(self, monkeypatch):
        """Configure cloud backend mode."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "browserbase")
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": "bb-123",
                "cdp_url": None,
                "features": {},
                "_first_nav": False,
            },
        )

    @pytest.fixture()
    def _hybrid_routing_enabled(self, monkeypatch):
        """Enable hybrid auto-local routing for private URLs."""
        monkeypatch.setattr(browser_tool, "_auto_local_for_private_urls", lambda: True)

    def test_blocks_aws_imds_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """AWS IMDS is blocked even when hybridRouting is enabled."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(browser_tool.browser_navigate(self.AWS_IMDS))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_blocks_aws_imds_v2_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """AWS IMDS v2 (169.254.169.254) is blocked."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(browser_tool.browser_navigate(self.AWS_IMDS_V2))

        assert result["success"] is False

    def test_blocks_gcp_imds_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """GCP IMDS (metadata.google.internal) is blocked."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(browser_tool.browser_navigate(self.GCP_IMDS))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_blocks_azure_imds_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """Azure IMDS (169.254.169.253) is blocked."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(browser_tool.browser_navigate(self.AZURE_IMDS))

        assert result["success"] is False

    def test_blocks_aliyun_imds_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """Aliyun IMDS (100.100.100.200) is blocked."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(browser_tool.browser_navigate(self.ALIYUN_IMDS))

        assert result["success"] is False

    def test_blocks_link_local_range_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """Entire 169.254.0.0/16 range is blocked."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        result = json.loads(
            browser_tool.browser_navigate("http://169.254.42.99/anything")
        )

        assert result["success"] is False

    def test_allows_public_url_with_hybrid_routing(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """Public URLs pass even with hybrid routing enabled."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(),
        )

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    def test_hybrid_routing_still_works_for_legitimate_private(
        self, monkeypatch, _cloud_mode, _hybrid_routing_enabled
    ):
        """Legitimate private URLs (non-IMDS) still route to local when allowed."""
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(),
        )

        result = json.loads(browser_tool.browser_navigate("http://192.168.1.1:8080/"))

        assert result["success"] is True
