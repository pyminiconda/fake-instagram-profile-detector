"""
ProfileFetcher -- Instagram profile data retrieval.

Strategy (in order):
  1. Check PROFILE_CACHE in SQLite -- return cached data if not expired.
  2. Try Instaloader (authenticated if .env credentials present).
  3. Fallback: scrape Instagram's public web page for profile JSON.

Falls back gracefully at every step so the app always has a chance to work.
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


class ProfileNotFoundError(Exception):
    """Raised when the Instagram profile does not exist."""
    pass


class RateLimitError(Exception):
    """Raised when Instagram rate-limits the request."""
    pass


class PrivateProfileError(Exception):
    """Raised when the profile is private."""
    pass


def _is_demo_mode() -> bool:
    """Check whether Instaloader credentials are configured."""
    username = os.getenv("INSTA_USERNAME", "").strip()
    password = os.getenv("INSTA_PASSWORD", "").strip()
    return not (username and password)


class ProfileFetcher:
    """Fetch Instagram profile data, with caching and multiple fallbacks."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.demo_mode = _is_demo_mode()
        self._loader = None

        if not self.demo_mode:
            self._init_instaloader()

    # ------------------------------------------------------------------
    # Instaloader session
    # ------------------------------------------------------------------
    def _init_instaloader(self):
        """Initialise and log in to an Instaloader session."""
        try:
            import instaloader
            self._loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
            )
            username = os.getenv("INSTA_USERNAME")
            password = os.getenv("INSTA_PASSWORD")
            self._loader.login(username, password)
            print("[INFO] Instaloader logged in successfully.")
        except Exception as exc:
            print(f"[WARNING] Instaloader login failed: {exc}")
            print("   Will use web scraping fallback.")
            self.demo_mode = True
            self._loader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_profile(self, username: str) -> dict:
        """
        Fetch profile data for *username*.

        Returns a dict with keys:
            username, followersCount, followingCount, postsCount,
            isPrivate, isVerified, hasProfilePicture, biography,
            externalUrl, fullName

        Raises:
            ProfileNotFoundError, PrivateProfileError, RateLimitError
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty.")

        username = username.strip().lower()

        # 1. Try cache first
        cached = self.db.get_cached_profile(username)
        if cached:
            return cached

        # 2. Try Instaloader (if authenticated)
        if self._loader is not None:
            try:
                result = self._fetch_via_instaloader(username)
                if result:
                    return result
            except (ProfileNotFoundError, PrivateProfileError):
                raise  # Re-raise these specific errors
            except Exception as exc:
                print(f"[WARNING] Instaloader fetch failed: {exc}")
                print("   Trying web scraping fallback...")

        # 3. Fallback: scrape Instagram's public web page
        return self._fetch_via_web(username)

    def _fetch_via_instaloader(self, username: str) -> dict:
        """Fetch profile via authenticated Instaloader."""
        import instaloader

        time.sleep(2)  # Rate-limit guard

        try:
            profile = instaloader.Profile.from_username(
                self._loader.context, username
            )
        except instaloader.exceptions.ProfileNotExistsException:
            raise ProfileNotFoundError(
                f"Profile '{username}' not found. Check the username and try again."
            )
        except instaloader.exceptions.ConnectionException as exc:
            raise RateLimitError(
                f"Instagram connection issue: {exc}"
            )
        except Exception as exc:
            raise RateLimitError(f"Failed to fetch profile: {exc}")

        if profile.is_private:
            raise PrivateProfileError(
                "This profile is private. Try using the Manual Entry tab instead."
            )

        profile_data = self._build_profile_dict(
            username=profile.username,
            followers=profile.followers,
            following=profile.followees,
            posts=profile.mediacount,
            is_private=profile.is_private,
            is_verified=profile.is_verified,
            has_pic=profile.profile_pic_url is not None,
            biography=profile.biography or "",
            external_url=profile.external_url or "",
            full_name=profile.full_name or "",
        )

        self.db.cache_profile(profile_data)
        return profile_data

    def _fetch_via_web(self, username: str) -> dict:
        """
        Fallback: fetch profile from Instagram's public web page.
        Parses the embedded JSON data or meta tags.
        """
        url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "X-IG-App-ID": "936619743392459",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://www.instagram.com/{username}/",
            "X-Requested-With": "XMLHttpRequest",
        }

        time.sleep(2)

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            user = data.get("data", {}).get("user")
            if user is None:
                raise ProfileNotFoundError(
                    f"Profile '{username}' not found. Check the username and try again."
                )

            is_private = user.get("is_private", False)
            if is_private:
                raise PrivateProfileError(
                    "This profile is private. Try using the Manual Entry tab instead."
                )

            profile_data = self._build_profile_dict(
                username=user.get("username", username),
                followers=user.get("edge_followed_by", {}).get("count", 0),
                following=user.get("edge_follow", {}).get("count", 0),
                posts=user.get("edge_owner_to_timeline_media", {}).get("count", 0),
                is_private=is_private,
                is_verified=user.get("is_verified", False),
                has_pic=user.get("profile_pic_url") is not None,
                biography=user.get("biography", ""),
                external_url=user.get("external_url", "") or "",
                full_name=user.get("full_name", ""),
            )

            self.db.cache_profile(profile_data)
            print(f"[OK] Fetched @{username} via web API.")
            return profile_data

        except (ProfileNotFoundError, PrivateProfileError):
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ProfileNotFoundError(
                    f"Profile '{username}' not found. Check the username and try again."
                )
            elif exc.code in (401, 403):
                # Try the HTML page scraping as last resort
                return self._fetch_via_html(username)
            else:
                raise RateLimitError(
                    f"Instagram returned HTTP {exc.code}. Try again later or use Manual Entry."
                )
        except Exception as exc:
            # Try HTML scraping as last resort
            print(f"[WARNING] Web API failed: {exc}. Trying HTML scrape...")
            return self._fetch_via_html(username)

    def _fetch_via_html(self, username: str) -> dict:
        """
        Last resort: scrape the public HTML page for meta/og tags
        to extract at least basic profile info.
        """
        url = f"https://www.instagram.com/{username}/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        time.sleep(2)

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")

            # Check for 'page not found' indicators
            if "Page Not Found" in html or "Sorry, this page" in html:
                raise ProfileNotFoundError(
                    f"Profile '{username}' not found. Check the username and try again."
                )

            # Try to extract shared_data JSON
            shared_data_match = re.search(
                r'window\._sharedData\s*=\s*({.+?});</script>', html
            )
            if shared_data_match:
                shared = json.loads(shared_data_match.group(1))
                user = (
                    shared.get("entry_data", {})
                    .get("ProfilePage", [{}])[0]
                    .get("graphql", {})
                    .get("user", {})
                )
                if user:
                    profile_data = self._build_profile_dict(
                        username=user.get("username", username),
                        followers=user.get("edge_followed_by", {}).get("count", 0),
                        following=user.get("edge_follow", {}).get("count", 0),
                        posts=user.get("edge_owner_to_timeline_media", {}).get("count", 0),
                        is_private=user.get("is_private", False),
                        is_verified=user.get("is_verified", False),
                        has_pic=user.get("profile_pic_url") is not None,
                        biography=user.get("biography", ""),
                        external_url=user.get("external_url", "") or "",
                        full_name=user.get("full_name", ""),
                    )
                    self.db.cache_profile(profile_data)
                    print(f"[OK] Fetched @{username} via HTML shared data.")
                    return profile_data

            # Try extracting from og:description meta tag
            # Format: "X Followers, Y Following, Z Posts - See Instagram photos..."
            og_match = re.search(
                r'<meta\s+(?:property="og:description"|content="([^"]*)")\s+'
                r'(?:property="og:description"|content="([^"]*)")',
                html,
            )
            desc = ""
            if og_match:
                desc = og_match.group(1) or og_match.group(2) or ""

            if not desc:
                # Try alternative pattern
                desc_match = re.search(
                    r'content="([\d,.]+[KkMm]?\s+Followers?,\s*[\d,.]+[KkMm]?\s+Following,\s*[\d,.]+[KkMm]?\s+Posts?[^"]*)"',
                    html,
                )
                if desc_match:
                    desc = desc_match.group(1)

            if desc:
                followers, following, posts = self._parse_og_description(desc)

                # Extract full name from title
                title_match = re.search(r'<title>([^(]*)\(', html)
                full_name = title_match.group(1).strip() if title_match else ""

                profile_data = self._build_profile_dict(
                    username=username,
                    followers=followers,
                    following=following,
                    posts=posts,
                    is_private=False,
                    is_verified=False,
                    has_pic=True,
                    biography="",
                    external_url="",
                    full_name=full_name,
                )
                self.db.cache_profile(profile_data)
                print(f"[OK] Fetched @{username} via HTML meta tags.")
                return profile_data

            # Absolute last resort: couldn't parse anything
            raise RateLimitError(
                "Instagram is blocking automated requests. "
                "Please try again later or use the Manual Entry tab."
            )

        except (ProfileNotFoundError, PrivateProfileError, RateLimitError):
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ProfileNotFoundError(
                    f"Profile '{username}' not found. Check the username and try again."
                )
            raise RateLimitError(
                f"Instagram returned HTTP {exc.code}. Try again later or use Manual Entry."
            )
        except Exception as exc:
            raise RateLimitError(
                f"Could not fetch profile data. Instagram may be blocking requests. "
                f"Please use the Manual Entry tab. (Error: {exc})"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_profile_dict(self, username, followers, following, posts,
                            is_private, is_verified, has_pic, biography,
                            external_url, full_name) -> dict:
        """Build a standardized profile dict."""
        return {
            "username": username,
            "followersCount": followers,
            "followingCount": following,
            "postsCount": posts,
            "isPrivate": is_private,
            "isVerified": is_verified,
            "hasProfilePicture": has_pic,
            "biography": biography,
            "externalUrl": external_url,
            "fullName": full_name,
        }

    def _parse_og_description(self, desc: str) -> tuple:
        """Parse followers/following/posts from og:description text."""
        def _parse_count(text: str) -> int:
            text = text.strip().replace(",", "")
            if text.upper().endswith("K"):
                return int(float(text[:-1]) * 1000)
            elif text.upper().endswith("M"):
                return int(float(text[:-1]) * 1000000)
            try:
                return int(text)
            except ValueError:
                return 0

        followers = following = posts = 0
        parts = desc.split(",")
        for part in parts:
            part = part.strip()
            if "Follower" in part:
                num = re.search(r'([\d,.]+[KkMm]?)', part)
                if num:
                    followers = _parse_count(num.group(1))
            elif "Following" in part:
                num = re.search(r'([\d,.]+[KkMm]?)', part)
                if num:
                    following = _parse_count(num.group(1))
            elif "Post" in part:
                num = re.search(r'([\d,.]+[KkMm]?)', part)
                if num:
                    posts = _parse_count(num.group(1))

        return followers, following, posts

    def is_demo_mode(self) -> bool:
        """Return True when running without Instagram credentials."""
        return self.demo_mode
