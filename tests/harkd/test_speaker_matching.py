"""Tests for speaker-to-profile matching logic."""

import numpy as np

from harkd.services.processing_worker import ProcessingWorker


class TestMatchSpeakersToProfiles:
    """Tests for _match_speakers_to_profiles static method."""

    def test_identical_embeddings_match(self):
        """Identical embeddings should produce cosine similarity of 1.0."""
        vec = [0.1, 0.2, 0.3, 0.4]
        speaker_embeddings = {"SPEAKER_00": vec}
        known_embeddings = {"Alice": (vec, "profile-abc")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert matches == {"SPEAKER_00": ("Alice", "profile-abc")}

    def test_below_threshold_no_match(self):
        """Embeddings below threshold should not match."""
        speaker_embeddings = {"SPEAKER_00": [1.0, 0.0, 0.0]}
        known_embeddings = {"Alice": ([0.0, 1.0, 0.0], "profile-abc")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert matches == {}

    def test_greedy_one_to_one_assignment(self):
        """Each anonymous speaker should match at most one profile, and vice versa."""
        # SPEAKER_00 is closest to Alice, SPEAKER_01 is closest to Bob
        speaker_embeddings = {
            "SPEAKER_00": [1.0, 0.0, 0.0],
            "SPEAKER_01": [0.0, 1.0, 0.0],
        }
        known_embeddings = {
            "Alice": ([0.9, 0.1, 0.0], "profile-a"),
            "Bob": ([0.1, 0.9, 0.0], "profile-b"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.5
        )

        assert len(matches) == 2
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")
        assert matches["SPEAKER_01"] == ("Bob", "profile-b")

    def test_no_double_matching(self):
        """Two speakers both close to same profile — only the best match wins."""
        speaker_embeddings = {
            "SPEAKER_00": [1.0, 0.0, 0.0],
            "SPEAKER_01": [0.95, 0.05, 0.0],  # Also close to Alice
        }
        known_embeddings = {
            "Alice": ([1.0, 0.0, 0.0], "profile-a"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.5
        )

        assert len(matches) == 1
        # SPEAKER_00 is an exact match, should win
        assert "SPEAKER_00" in matches
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")

    def test_empty_speaker_embeddings(self):
        """Empty speaker embeddings returns empty result."""
        matches = ProcessingWorker._match_speakers_to_profiles(
            {}, {"Alice": ([0.1, 0.2], "profile-a")}, threshold=0.7
        )
        assert matches == {}

    def test_empty_known_embeddings(self):
        """Empty known embeddings returns empty result."""
        matches = ProcessingWorker._match_speakers_to_profiles(
            {"SPEAKER_00": [0.1, 0.2]}, {}, threshold=0.7
        )
        assert matches == {}

    def test_both_empty(self):
        """Both empty returns empty result."""
        matches = ProcessingWorker._match_speakers_to_profiles({}, {}, threshold=0.7)
        assert matches == {}

    def test_zero_norm_vector_no_crash(self):
        """Zero-norm vectors should not crash and should not match."""
        speaker_embeddings = {"SPEAKER_00": [0.0, 0.0, 0.0]}
        known_embeddings = {"Alice": ([0.1, 0.2, 0.3], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.5
        )

        assert matches == {}

    def test_zero_norm_known_embedding(self):
        """Zero-norm known embedding should not match."""
        speaker_embeddings = {"SPEAKER_00": [0.1, 0.2, 0.3]}
        known_embeddings = {"Alice": ([0.0, 0.0, 0.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.5
        )

        assert matches == {}

    def test_more_speakers_than_profiles(self):
        """Unmatched speakers should keep their anonymous labels."""
        speaker_embeddings = {
            "SPEAKER_00": [1.0, 0.0, 0.0],
            "SPEAKER_01": [0.0, 1.0, 0.0],
            "SPEAKER_02": [0.0, 0.0, 1.0],
        }
        known_embeddings = {
            "Alice": ([1.0, 0.0, 0.0], "profile-a"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert len(matches) == 1
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")
        assert "SPEAKER_01" not in matches
        assert "SPEAKER_02" not in matches

    def test_more_profiles_than_speakers(self):
        """Unused profiles should be ignored."""
        speaker_embeddings = {
            "SPEAKER_00": [1.0, 0.0, 0.0],
        }
        known_embeddings = {
            "Alice": ([1.0, 0.0, 0.0], "profile-a"),
            "Bob": ([0.0, 1.0, 0.0], "profile-b"),
            "Charlie": ([0.0, 0.0, 1.0], "profile-c"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert len(matches) == 1
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")

    def test_dimension_mismatch_returns_empty(self):
        """Mismatched embedding dimensions should return empty (not crash)."""
        speaker_embeddings = {"SPEAKER_00": [0.1, 0.2, 0.3]}
        known_embeddings = {"Alice": ([0.1, 0.2, 0.3, 0.4, 0.5], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert matches == {}

    def test_threshold_zero_matches_everything(self):
        """Threshold of 0.0 should match any non-zero-norm vectors."""
        speaker_embeddings = {"SPEAKER_00": [1.0, 0.0, 0.0]}
        known_embeddings = {"Alice": ([0.0, 0.0, 1.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.0
        )

        assert len(matches) == 1

    def test_threshold_one_requires_exact_match(self):
        """Threshold of 1.0 should only match exact vectors."""
        speaker_embeddings = {"SPEAKER_00": [0.9, 0.1, 0.0]}
        known_embeddings = {"Alice": ([1.0, 0.0, 0.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=1.0
        )

        # Cosine sim of [0.9, 0.1, 0] and [1, 0, 0] is ~0.994, not 1.0
        assert matches == {}

    def test_high_dimensional_vectors(self):
        """Test with 512-dim vectors (realistic ECAPA-VOXCELEB size)."""
        rng = np.random.default_rng(42)
        base_vec = rng.standard_normal(512)
        # Create a similar vector (slight perturbation)
        similar_vec = base_vec + rng.standard_normal(512) * 0.1

        speaker_embeddings = {"SPEAKER_00": base_vec.tolist()}
        known_embeddings = {"Alice": (similar_vec.tolist(), "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        # Should match — vectors are very similar
        assert len(matches) == 1
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")


class TestMatchSpeakersEdgeCases:
    """Edge cases and potential bug areas in _match_speakers_to_profiles."""

    def test_negative_cosine_similarity_not_matched_at_threshold_zero(self):
        """Opposing vectors (cosine sim = -1) should NOT match even at threshold=0."""
        speaker_embeddings = {"SPEAKER_00": [1.0, 0.0, 0.0]}
        known_embeddings = {"Alice": ([-1.0, 0.0, 0.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.0
        )

        # cosine sim = -1.0, which is < 0.0 threshold
        assert matches == {}

    def test_ragged_vectors_within_speaker_embeddings(self):
        """Different-length vectors in speaker_embeddings should not crash."""
        speaker_embeddings = {
            "SPEAKER_00": [0.1, 0.2],
            "SPEAKER_01": [0.3, 0.4, 0.5],
        }
        known_embeddings = {"Alice": ([0.1, 0.2, 0.3], "profile-a")}

        # numpy will raise ValueError on ragged arrays — should be caught
        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert matches == {}

    def test_ragged_vectors_within_known_embeddings(self):
        """Different-length vectors in known_embeddings should not crash."""
        speaker_embeddings = {"SPEAKER_00": [0.1, 0.2, 0.3]}
        known_embeddings = {
            "Alice": ([0.1, 0.2], "profile-a"),
            "Bob": ([0.3, 0.4, 0.5], "profile-b"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        assert matches == {}

    def test_empty_vector_in_speaker_embeddings(self):
        """Empty vector [] in speaker embeddings should not crash."""
        speaker_embeddings = {"SPEAKER_00": []}
        known_embeddings = {"Alice": ([0.1, 0.2, 0.3], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        # Dimension mismatch (0 vs 3) → empty
        assert matches == {}

    def test_greedy_conflict_second_best_wins_for_displaced_speaker(self):
        """When greedy picks best match for one speaker, another can still match elsewhere.

        Scenario:
        - SPEAKER_00: very close to Alice (sim=0.99), somewhat close to Bob (sim=0.8)
        - SPEAKER_01: close to Alice (sim=0.95), close to Bob (sim=0.85)

        Greedy should assign SPEAKER_00→Alice (0.99), then SPEAKER_01→Bob (0.85).
        SPEAKER_01 should NOT get Alice (already taken) even though sim=0.95 > 0.85.
        """
        # Create vectors where SPEAKER_00 is closest to Alice, SPEAKER_01 second-closest
        alice_vec = np.array([1.0, 0.0, 0.0])
        bob_vec = np.array([0.0, 1.0, 0.0])

        # SPEAKER_00: very aligned with Alice
        spk0 = np.array([0.99, 0.01, 0.0])
        # SPEAKER_01: closer to Alice than Bob, but Alice will be taken
        spk1 = np.array([0.7, 0.7, 0.0])

        speaker_embeddings = {
            "SPEAKER_00": spk0.tolist(),
            "SPEAKER_01": spk1.tolist(),
        }
        known_embeddings = {
            "Alice": (alice_vec.tolist(), "profile-a"),
            "Bob": (bob_vec.tolist(), "profile-b"),
        }

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.5
        )

        assert len(matches) == 2
        assert matches["SPEAKER_00"] == ("Alice", "profile-a")
        assert matches["SPEAKER_01"] == ("Bob", "profile-b")

    def test_both_zero_norm_vectors(self):
        """Both speaker and profile are zero vectors — should not match or crash."""
        speaker_embeddings = {"SPEAKER_00": [0.0, 0.0, 0.0]}
        known_embeddings = {"Alice": ([0.0, 0.0, 0.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.0
        )

        assert matches == {}

    def test_single_speaker_single_profile(self):
        """Minimal case: 1×1 matrix should work correctly."""
        speaker_embeddings = {"SPEAKER_00": [0.5, 0.5]}
        known_embeddings = {"Alice": ([0.5, 0.5], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.9
        )

        assert matches == {"SPEAKER_00": ("Alice", "profile-a")}

    def test_cosine_similarity_is_direction_not_magnitude(self):
        """Vectors with same direction but different magnitude should still match."""
        speaker_embeddings = {"SPEAKER_00": [0.001, 0.001, 0.001]}
        known_embeddings = {"Alice": ([100.0, 100.0, 100.0], "profile-a")}

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.99
        )

        # Same direction — cosine sim = 1.0 regardless of magnitude
        assert matches == {"SPEAKER_00": ("Alice", "profile-a")}

    def test_many_speakers_many_profiles(self):
        """Test with 10 speakers and 10 profiles (realistic meeting size)."""
        rng = np.random.default_rng(99)
        n = 10
        dim = 512

        # Generate n distinct unit vectors
        base_vectors = []
        for _ in range(n):
            v = rng.standard_normal(dim)
            v = v / np.linalg.norm(v)
            base_vectors.append(v)

        # Speaker embeddings: very slight perturbation of base vectors
        # (In 512d, even small noise reduces cosine sim significantly,
        # so use 0.01 stddev to stay well above the 0.7 threshold)
        speaker_embeddings = {}
        for i in range(n):
            perturbed = base_vectors[i] + rng.standard_normal(dim) * 0.01
            speaker_embeddings[f"SPEAKER_{i:02d}"] = perturbed.tolist()

        # Known embeddings: same base vectors
        known_embeddings = {}
        for i in range(n):
            known_embeddings[f"Person_{i}"] = (base_vectors[i].tolist(), f"profile-{i}")

        matches = ProcessingWorker._match_speakers_to_profiles(
            speaker_embeddings, known_embeddings, threshold=0.7
        )

        # All should match — vectors are very close
        assert len(matches) == n
        for i in range(n):
            label = f"SPEAKER_{i:02d}"
            assert label in matches
            assert matches[label] == (f"Person_{i}", f"profile-{i}")
