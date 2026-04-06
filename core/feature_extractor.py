"""
FeatureExtractor — Computes the 7 derived features used by the ML model.

Works with both:
  • Live Instaloader profile dicts (from ProfileFetcher)
  • Raw dataset rows (pandas Series from the InstaFake CSV)
"""

import re
import math
from typing import List, Dict, Union


# Feature names in the order expected by the model
FEATURE_NAMES = [
    "followerRatio",
    "profileCompleteness",
    "engagementRate",
    "bioLength",
    "usernameAnomalyScore",
    "postFrequency",
    "hasPicture",
]


class FeatureExtractor:
    """Compute the 7-feature vector from profile data."""

    # ------------------------------------------------------------------
    # Individual feature calculators
    # ------------------------------------------------------------------
    @staticmethod
    def calc_follower_following_ratio(followers: int, following: int) -> float:
        """Ratio of followers to following. Higher = more organic."""
        return followers / (following + 1)

    @staticmethod
    def calc_profile_completeness(has_pic: bool, bio_length: int,
                                  has_url: bool = False,
                                  full_name: str = "") -> float:
        """Score 0-1 representing how 'complete' the profile is."""
        score = 0.0
        if has_pic:
            score += 0.35
        if bio_length > 0:
            score += 0.25
        if bio_length > 30:
            score += 0.10  # bonus for substantial bio
        if has_url:
            score += 0.15
        if full_name and len(full_name.strip()) > 0:
            score += 0.15
        return min(score, 1.0)

    @staticmethod
    def calc_engagement_rate(posts: int, followers: int) -> float:
        """Proxy engagement rate (posts per follower). Dataset lacks likes/comments."""
        return posts / (followers + 1)

    @staticmethod
    def analyze_bio(bio_text: str) -> Dict:
        """Analyse biography text and return metadata dict."""
        if not bio_text:
            bio_text = ""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        spam_keywords = [
            "follow", "dm", "promo", "free", "click", "link",
            "earn", "money", "giveaway", "win",
        ]
        bio_lower = bio_text.lower()
        return {
            "length": len(bio_text),
            "emoji_count": len(emoji_pattern.findall(bio_text)),
            "keyword_flags": {kw: kw in bio_lower for kw in spam_keywords},
        }

    @staticmethod
    def calc_username_anomaly_score(username: str) -> float:
        """Score reflecting how 'bot-like' a username looks. Higher = more suspicious."""
        if not username:
            return 1.0
        digit_count = sum(c.isdigit() for c in username)
        special_count = sum(not c.isalnum() and c != "_" for c in username)
        length = len(username)

        digit_ratio = digit_count / (length + 1)
        special_ratio = special_count / (length + 1)
        length_penalty = 0.0
        if length > 20:
            length_penalty = 0.2
        elif length <= 3:
            length_penalty = 0.15

        score = digit_ratio * 0.5 + special_ratio * 0.3 + length_penalty
        return min(score, 1.0)

    @staticmethod
    def calc_post_frequency(posts: int) -> float:
        """Post count used as a proxy for post frequency (no account age in dataset)."""
        return float(posts)

    @staticmethod
    def calc_profile_pic_presence(has_pic) -> float:
        """Returns 1.0 if profile picture exists, else 0.0."""
        return 1.0 if has_pic else 0.0

    # ------------------------------------------------------------------
    # Unified extraction
    # ------------------------------------------------------------------
    def extract_from_profile(self, profile: dict) -> List[float]:
        """
        Extract feature vector from a live Instaloader profile dict.

        Expected keys: username, followersCount, followingCount, postsCount,
                       hasProfilePicture, biography, externalUrl, fullName
        """
        followers = int(profile.get("followersCount", 0))
        following = int(profile.get("followingCount", 0))
        posts = int(profile.get("postsCount", 0))
        has_pic = bool(profile.get("hasProfilePicture", False))
        bio = str(profile.get("biography", ""))
        has_url = bool(profile.get("externalUrl", ""))
        full_name = str(profile.get("fullName", ""))
        username = str(profile.get("username", ""))

        bio_info = self.analyze_bio(bio)

        return [
            self.calc_follower_following_ratio(followers, following),
            self.calc_profile_completeness(has_pic, bio_info["length"], has_url, full_name),
            self.calc_engagement_rate(posts, followers),
            float(bio_info["length"]),
            self.calc_username_anomaly_score(username),
            self.calc_post_frequency(posts),
            self.calc_profile_pic_presence(has_pic),
        ]

    def extract_from_dataset_row(self, row) -> List[float]:
        """
        Extract feature vector from a raw dataset row (pandas Series).

        Expected columns: user_follower_count, user_following_count,
                          user_media_count, user_has_profil_pic,
                          user_biography_length, username_length,
                          username_digit_count
        """
        followers = int(row.get("user_follower_count", 0))
        following = int(row.get("user_following_count", 0))
        posts = int(row.get("user_media_count", 0))
        has_pic = bool(row.get("user_has_profil_pic", 0))
        bio_length = int(row.get("user_biography_length", 0))
        uname_length = int(row.get("username_length", 0))
        uname_digits = int(row.get("username_digit_count", 0))

        # Approximate profile completeness from available columns
        completeness = 0.0
        if has_pic:
            completeness += 0.35
        if bio_length > 0:
            completeness += 0.25
        if bio_length > 30:
            completeness += 0.10
        completeness = min(completeness + 0.15, 1.0)  # assume name present

        # Username anomaly from digit ratio + length penalty
        digit_ratio = uname_digits / (uname_length + 1)
        length_penalty = 0.2 if uname_length > 20 else (0.15 if uname_length <= 3 else 0.0)
        anomaly = min(digit_ratio * 0.5 + length_penalty, 1.0)

        return [
            self.calc_follower_following_ratio(followers, following),
            completeness,
            self.calc_engagement_rate(posts, followers),
            float(bio_length),
            anomaly,
            self.calc_post_frequency(posts),
            self.calc_profile_pic_presence(has_pic),
        ]

    def extract_from_manual_input(self, followers: int, following: int,
                                   posts: int, has_pic: bool,
                                   bio_length: int, username: str,
                                   has_url: bool = False,
                                   full_name: str = "") -> List[float]:
        """
        Extract features from manually entered values (demo mode).
        """
        bio_text = "x" * bio_length  # stub for length-based analysis

        return [
            self.calc_follower_following_ratio(followers, following),
            self.calc_profile_completeness(has_pic, bio_length, has_url, full_name),
            self.calc_engagement_rate(posts, followers),
            float(bio_length),
            self.calc_username_anomaly_score(username),
            self.calc_post_frequency(posts),
            self.calc_profile_pic_presence(has_pic),
        ]
