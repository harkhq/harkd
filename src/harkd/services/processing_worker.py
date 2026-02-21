"""Queue-based processing worker for recordings."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from harkd.config import HarkdSettings
from harkd.events import Event, get_event_bus
from harkd.exceptions import RecordingNotFoundError, RetryNotAllowedError
from harkd.services.title_generator import generate_title
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.transcription import LocalBackend, TranscriptionRequest, create_backend

if TYPE_CHECKING:
    from harkd.services.voice_profile_service import VoiceProfileService

__all__ = ["ProcessingWorker"]

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """Queue-based worker for processing recordings sequentially.

    Processes one recording at a time from an asyncio.Queue. Handles retry
    logic with exponential backoff for transient errors and classifies
    errors as transient vs permanent to avoid futile retries.

    Designed to be started in the FastAPI lifespan and shared across requests.
    """

    def __init__(
        self,
        storage: FilesystemRecordingStorage,
        config: HarkdSettings,
        voice_profile_service: VoiceProfileService | None = None,
    ):
        self._storage = storage
        self._config = config
        self._backend = create_backend(config)
        self._voice_profile_service = voice_profile_service
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._active: set[str] = set()  # IDs in queue or being processed
        self._current_id: str | None = None  # ID currently being processed
        self._task: asyncio.Task[None] | None = None
        self._shutdown = False

    def start(self) -> None:
        """Start the worker loop as a background task."""
        self._shutdown = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Processing worker started (backend=%s)",
            self._config.transcription.backend,
        )

    async def pre_warm_backend(self) -> None:
        """Pre-warm the transcription backend (e.g. start provisioning infra)."""
        if hasattr(self._backend, "pre_warm"):
            try:
                await self._backend.pre_warm()
            except Exception:
                logger.warning("Backend pre-warm failed", exc_info=True)

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Graceful shutdown: finish current item, discard queue."""
        logger.info("Processing worker shutting down...")
        self._shutdown = True
        # Put sentinel to unblock queue.get()
        await self._queue.put("")
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.CancelledError, TimeoutError):
                self._task.cancel()
        await self._backend.close()

    def enqueue(self, recording_id: str) -> bool:
        """Add a recording to the processing queue.

        Returns False if already queued/processing (dedup).
        """
        if recording_id in self._active:
            logger.debug(f"[{recording_id}] Already queued, skipping enqueue")
            return False
        self._active.add(recording_id)
        self._queue.put_nowait(recording_id)
        logger.info(f"[{recording_id}] Enqueued for processing (queue_size={self._queue.qsize()})")
        return True

    async def recover(self) -> int:
        """Scan storage for stuck/retryable recordings on startup.

        Returns count of recordings enqueued.
        """
        count = 0
        all_recordings = await self._storage.list(limit=0, offset=0)
        for rec in all_recordings:
            if rec.status == "processing":
                # Stuck from crash — increment retry and re-enqueue if allowed
                rec.retry_count += 1
                rec.last_error = "Daemon restarted during processing"
                rec.last_error_at = datetime.now(UTC)
                rec.error_history.append(
                    {
                        "attempt": rec.retry_count,
                        "error": "Daemon restarted during processing",
                        "transient": True,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                if rec.retry_count <= rec.max_retries:
                    await self._storage.update(rec)
                    self.enqueue(rec.id)
                    count += 1
                    logger.info(
                        f"[{rec.id}] Recovered stuck processing recording "
                        f"(attempt {rec.retry_count}/{rec.max_retries})"
                    )
                else:
                    rec.status = "error"
                    await self._storage.update(rec)
                    logger.warning(
                        f"[{rec.id}] Stuck processing recording exceeded max retries, "
                        "marked as error"
                    )
            elif rec.status == "error" and rec.retry_count < rec.max_retries:
                if self._is_transient_error(rec.last_error):
                    self.enqueue(rec.id)
                    count += 1
                    logger.info(
                        f"[{rec.id}] Re-enqueued transient error recording "
                        f"(attempt {rec.retry_count}/{rec.max_retries})"
                    )
        return count

    async def retry(self, recording_id: str) -> None:
        """Manual retry or reprocess. Validates state and enqueues.

        Raises:
            RecordingNotFoundError: If recording doesn't exist
            RetryNotAllowedError: If recording is not in error/complete state or already queued
        """
        rec = await self._storage.get(recording_id)
        if rec is None:
            raise RecordingNotFoundError(recording_id)
        if rec.status not in ("error", "complete"):
            raise RetryNotAllowedError(
                recording_id, f"status is '{rec.status}', must be 'error' or 'complete'"
            )
        if recording_id in self._active:
            raise RetryNotAllowedError(recording_id, "already queued for processing")

        # Clear processing output when reprocessing a completed recording
        if rec.status == "complete":
            rec.model = None
            rec.language = None
            rec.language_confidence = None
            rec.diarized = None
            rec.speakers = []
            rec.speaker_embeddings = None
            rec.speaker_profiles = None
            rec.segments = []
            rec.transcript = None
            rec.processing_duration = None
            rec.executive_summary = []
            rec.meeting_notes = []
            rec.tasks = []
            rec.decisions = []

        # Reset retry/error state (for both error and complete)
        rec.retry_count = 0
        rec.last_error = None
        rec.last_error_at = None
        rec.error_history = []

        # Set status to processing and enqueue
        rec.status = "processing"
        rec.processing_stage = "preprocessing"
        rec.processing_progress = 0.0
        rec.processing_started_at = datetime.now(UTC)
        await self._storage.update(rec)
        self.enqueue(recording_id)
        get_event_bus().emit(Event("invalidate", {"entity": "recordings"}))

    async def _run_loop(self) -> None:
        """Main worker loop — process one item at a time."""
        while not self._shutdown:
            recording_id = await self._queue.get()
            if not recording_id or self._shutdown:
                break
            self._current_id = recording_id
            logger.info(f"[{recording_id}] Picked up for processing")
            try:
                await self._process_recording(recording_id)
            except Exception:
                logger.error(f"Unhandled error processing {recording_id}", exc_info=True)
            finally:
                self._current_id = None
                self._active.discard(recording_id)

    async def _process_recording(self, recording_id: str) -> None:
        """Process a single recording with retry logic."""
        _should_reenqueue = False
        _retry_count = 0
        _start_time = time.monotonic()

        try:
            storage_recording = await self._storage.get(recording_id)
            if storage_recording is None:
                logger.error(f"[{recording_id}] Recording not found during processing")
                return

            _retry_count = storage_recording.retry_count

            # Ensure status is processing
            if storage_recording.status != "processing":
                storage_recording.status = "processing"
                storage_recording.processing_stage = "preprocessing"
                storage_recording.processing_progress = 0.0
                await self._storage.update(storage_recording)

            # Audio file is stored alongside metadata
            audio_path = self._storage.base_path / "recordings" / recording_id / "audio.wav"
            if not await asyncio.to_thread(audio_path.exists):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            settings = storage_recording.settings
            recording_cfg = self._config.recording

            # Compute dynamic timeout based on audio duration
            audio_duration = storage_recording.duration or 0.0
            dynamic_timeout = max(
                recording_cfg.transcription_timeout,
                int(audio_duration * 0.5),
            )
            logger.info(
                f"[{recording_id}] Audio duration={audio_duration:.0f}s, timeout={dynamic_timeout}s"
            )

            # Stage 1: Transcription
            logger.info(f"[{recording_id}] Stage 1: Transcription")
            storage_recording.processing_stage = "transcription"
            storage_recording.processing_progress = 0.0
            await self._storage.update(storage_recording)
            get_event_bus().emit(
                Event(
                    "processing_update",
                    {
                        "recording_id": recording_id,
                        "status": "processing",
                        "processing_stage": "transcription",
                        "processing_progress": 0.0,
                    },
                )
            )

            diarization_enabled = settings.get("diarization", False)
            word_timestamps = settings.get("word_timestamps", False)
            model_name = settings.get("model", "base")
            language = settings.get("language") if settings.get("language") != "auto" else None

            # Diarization tuning params from per-recording settings
            num_speakers = settings.get("num_speakers")
            min_speakers = settings.get("min_speakers")
            max_speakers = settings.get("max_speakers")
            clustering_threshold = settings.get("clustering_threshold")

            if diarization_enabled:
                hf_token = self._config.hf_token
                logger.info(f"[{recording_id}] Using diarization")
                storage_recording.processing_stage = "diarization"
                storage_recording.processing_progress = 0.3
                await self._storage.update(storage_recording)
                get_event_bus().emit(
                    Event(
                        "processing_update",
                        {
                            "recording_id": recording_id,
                            "status": "processing",
                            "processing_stage": "diarization",
                            "processing_progress": 0.3,
                        },
                    )
                )
            else:
                hf_token = None

            result_dict = await self._run_transcription(
                recording_id,
                audio_path=audio_path,
                model_name=model_name,
                language=language,
                word_timestamps=True if diarization_enabled else word_timestamps,
                diarize=diarization_enabled,
                hf_token=hf_token,
                beam_size=recording_cfg.beam_size,
                batch_size=recording_cfg.batch_size,
                vad_onset=recording_cfg.vad_onset,
                vad_offset=recording_cfg.vad_offset,
                vad_method=recording_cfg.vad_method,
                timeout=dynamic_timeout,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                clustering_threshold=clustering_threshold,
            )

            segments, detected_language, language_confidence, speakers, speaker_embeddings = (
                self._convert_transcription_result(result_dict)
            )

            # Match speakers to known voice profiles
            speaker_profiles: dict[str, str] = {}
            if diarization_enabled and speaker_embeddings and self._voice_profile_service:
                try:
                    known = await self._voice_profile_service.get_known_embeddings()
                    if known:
                        speaker_matches = self._match_speakers_to_profiles(
                            speaker_embeddings, known, recording_cfg.speaker_match_threshold
                        )
                        if speaker_matches:
                            # Rename speakers in segments
                            for seg in segments:
                                if seg.get("speaker") in speaker_matches:
                                    seg["speaker"] = speaker_matches[seg["speaker"]][0]
                                for word in seg.get("words", []):
                                    if word.get("speaker") in speaker_matches:
                                        word["speaker"] = speaker_matches[word["speaker"]][0]

                            # Update speakers list (preserve order, deduplicate)
                            speakers = list(
                                dict.fromkeys(
                                    speaker_matches[s][0] if s in speaker_matches else s
                                    for s in speakers
                                )
                            )

                            # Set speaker_profiles mapping and auto-add embeddings
                            for anon_label, (
                                profile_name,
                                profile_id,
                            ) in speaker_matches.items():
                                speaker_profiles[profile_name] = profile_id
                                if anon_label in speaker_embeddings:
                                    await self._voice_profile_service.add_embedding(
                                        profile_id=profile_id,
                                        recording_id=recording_id,
                                        speaker_label=profile_name,
                                        vector=speaker_embeddings[anon_label],
                                        audio_duration=audio_duration,
                                    )

                            matched_names = {k: v[0] for k, v in speaker_matches.items()}
                            logger.info(
                                f"[{recording_id}] Matched {len(speaker_matches)} "
                                f"speakers to voice profiles: {matched_names}"
                            )
                except Exception:
                    logger.warning(
                        f"[{recording_id}] Voice profile matching failed, proceeding without",
                        exc_info=True,
                    )

            # Strip word-level timestamps if not requested by user
            if not word_timestamps:
                for seg in segments:
                    seg.pop("words", None)

            transcript = " ".join(seg["text"] for seg in segments)

            # Re-read title from storage — user may have updated via PATCH
            latest = await self._storage.get(recording_id)
            if latest is not None:
                storage_recording.title = latest.title

            # Generate title if not provided
            if storage_recording.title == "Untitled Recording":
                storage_recording.title = generate_title(transcript)
                logger.debug(f"[{recording_id}] Generated title: {storage_recording.title}")

            # Meeting minutes (if LLM enabled and transcript non-empty)
            if self._config.llm.enabled and transcript.strip():
                storage_recording.processing_stage = "meeting_minutes"
                storage_recording.processing_progress = 0.8
                await self._storage.update(storage_recording)
                get_event_bus().emit(
                    Event(
                        "processing_update",
                        {
                            "recording_id": recording_id,
                            "status": "processing",
                            "processing_stage": "meeting_minutes",
                            "processing_progress": 0.8,
                        },
                    )
                )

                try:
                    from harkd.llm.client import LLMClient

                    llm = LLMClient(self._config.llm)
                    minutes = await llm.generate_meeting_minutes(
                        transcript=transcript,
                        speakers=speakers,
                        language=detected_language,
                    )
                    storage_recording.executive_summary = minutes.executive_summary
                    storage_recording.meeting_notes = minutes.meeting_notes
                    storage_recording.tasks = minutes.tasks
                    storage_recording.decisions = minutes.decisions
                except Exception as e:
                    logger.warning(
                        f"[{recording_id}] Meeting minutes generation failed: {e}",
                        exc_info=True,
                    )

            # Update recording with final data
            processing_duration = time.monotonic() - _start_time
            storage_recording.status = "complete"
            storage_recording.processing_stage = None
            storage_recording.processing_progress = None
            storage_recording.processing_duration = round(processing_duration, 1)
            storage_recording.model = settings.get("model")
            storage_recording.language = detected_language
            storage_recording.language_confidence = language_confidence
            storage_recording.diarized = diarization_enabled
            storage_recording.speakers = speakers
            storage_recording.speaker_embeddings = speaker_embeddings
            storage_recording.speaker_profiles = speaker_profiles or None
            storage_recording.segments = segments
            storage_recording.transcript = transcript

            await self._storage.update(storage_recording)

            bus = get_event_bus()
            bus.emit(
                Event(
                    "recording_complete",
                    {
                        "recording_id": recording_id,
                        "title": storage_recording.title,
                    },
                )
            )
            bus.emit(Event("invalidate", {"entity": "recordings"}))
            if diarization_enabled:
                bus.emit(Event("invalidate", {"entity": "unassigned_speakers"}))

            logger.info(
                f"[{recording_id}] Processing complete in {processing_duration:.1f}s: "
                f"{len(segments)} segments, {len(speakers)} speakers, "
                f"language={detected_language}"
            )

        except asyncio.CancelledError:
            logger.info(f"[{recording_id}] Processing cancelled")
            raise
        except Exception as e:
            error_msg = str(e)
            is_transient = self._is_transient_error(error_msg)

            logger.error(
                f"[{recording_id}] Processing failed (transient={is_transient}): {e}",
                exc_info=True,
            )

            try:
                rec = await self._storage.get(recording_id)
                if rec:
                    rec.retry_count += 1
                    _retry_count = rec.retry_count
                    rec.last_error = error_msg
                    rec.last_error_at = datetime.now(UTC)
                    rec.error_history.append(
                        {
                            "attempt": rec.retry_count,
                            "error": error_msg,
                            "transient": is_transient,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    if is_transient and rec.retry_count < rec.max_retries:
                        # Stay in "processing" — will be re-enqueued in finally
                        _should_reenqueue = True
                    else:
                        rec.status = "error"
                        rec.processing_stage = None
                        rec.processing_progress = None
                    await self._storage.update(rec)
                    if rec.status == "error":
                        bus = get_event_bus()
                        bus.emit(
                            Event(
                                "recording_error",
                                {"recording_id": recording_id, "error": error_msg},
                            )
                        )
                        bus.emit(Event("invalidate", {"entity": "recordings"}))
            except Exception as update_error:
                logger.error(
                    f"[{recording_id}] Failed to update error status: {update_error}",
                    exc_info=True,
                )
        finally:
            if _should_reenqueue:
                # Keep in _active set, re-add to queue after backoff delay
                delay = min(30 * (2 ** (_retry_count - 1)), 300)
                loop = asyncio.get_event_loop()
                loop.call_later(
                    delay,
                    lambda rid=recording_id: self._queue.put_nowait(rid),
                )
            else:
                self._active.discard(recording_id)

    async def _run_transcription(
        self,
        recording_id: str,
        *,
        audio_path: Path,
        model_name: str,
        language: str | None,
        word_timestamps: bool,
        diarize: bool = False,
        hf_token: str | None = None,
        beam_size: int = 3,
        batch_size: int = 16,
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        vad_method: str = "pyannote",
        timeout: int = 1800,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        clustering_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run transcription via the configured backend, with fallback."""
        request = TranscriptionRequest(
            audio_path=audio_path,
            model_name=model_name,
            language=language,
            word_timestamps=word_timestamps,
            diarize=diarize,
            hf_token=hf_token,
            beam_size=beam_size,
            batch_size=batch_size,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            vad_method=vad_method,
            timeout=timeout,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold,
        )
        try:
            return await self._backend.transcribe(request)
        except Exception as e:
            if (
                not isinstance(self._backend, LocalBackend)
                and self._config.transcription.fallback_to_local
            ):
                logger.warning(
                    f"[{recording_id}] Remote transcription failed, falling back to local: {e}"
                )
                local = LocalBackend()
                return await local.transcribe(request)
            raise

    @staticmethod
    def _convert_transcription_result(
        result_dict: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str, float, list[str], dict[str, list[float]] | None]:
        """Convert transcription result to (segments, lang, confidence, speakers, embeddings)."""
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker"),
                "words": [
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"],
                        "speaker": w.get("speaker"),
                    }
                    for w in seg.get("words", [])
                ],
            }
            for seg in result_dict.get("segments", [])
        ]

        detected_language = result_dict.get("language", "unknown")
        language_confidence = result_dict.get("language_probability", 0.0)
        speakers = result_dict.get("speakers", [])
        speaker_embeddings = result_dict.get("speaker_embeddings")

        return segments, detected_language, language_confidence, speakers, speaker_embeddings

    @staticmethod
    def _match_speakers_to_profiles(
        speaker_embeddings: dict[str, list[float]],
        known_embeddings: dict[str, tuple[list[float], str]],
        threshold: float,
    ) -> dict[str, tuple[str, str]]:
        """Match anonymous speaker labels to known voice profiles via cosine similarity.

        Args:
            speaker_embeddings: Anon label -> embedding vector from diarization
            known_embeddings: Profile name -> (avg_embedding, profile_id)
            threshold: Minimum cosine similarity to accept a match

        Returns:
            Dict mapping anon label -> (profile_name, profile_id) for matches
        """
        if not speaker_embeddings or not known_embeddings:
            return {}

        anon_labels = list(speaker_embeddings.keys())
        profile_names = list(known_embeddings.keys())

        try:
            anon_matrix = np.array(
                [speaker_embeddings[label] for label in anon_labels], dtype=np.float64
            )
            prof_matrix = np.array(
                [known_embeddings[n][0] for n in profile_names], dtype=np.float64
            )
        except ValueError:
            logger.warning("Speaker embedding dimension mismatch, skipping profile matching")
            return {}

        if anon_matrix.shape[1] != prof_matrix.shape[1]:
            logger.warning(
                "Embedding dimension mismatch: speakers=%d, profiles=%d",
                anon_matrix.shape[1],
                prof_matrix.shape[1],
            )
            return {}

        # Compute norms and handle zero-norm vectors
        anon_norms = np.linalg.norm(anon_matrix, axis=1, keepdims=True)
        prof_norms = np.linalg.norm(prof_matrix, axis=1, keepdims=True)

        # Replace zero norms with 1 to avoid division by zero (similarity will be 0)
        anon_norms = np.where(anon_norms == 0, 1.0, anon_norms)
        prof_norms = np.where(prof_norms == 0, 1.0, prof_norms)

        anon_normed = anon_matrix / anon_norms
        prof_normed = prof_matrix / prof_norms

        # Cosine similarity matrix: (num_anon, num_profiles)
        sim_matrix = anon_normed @ prof_normed.T

        # Force similarity to -inf for zero-norm vectors so they never match
        # (0.0 would still match at threshold=0.0 since 0.0 < 0.0 is False)
        zero_anon = np.linalg.norm(anon_matrix, axis=1) == 0
        zero_prof = np.linalg.norm(prof_matrix, axis=1) == 0
        sim_matrix[zero_anon, :] = -np.inf
        sim_matrix[:, zero_prof] = -np.inf

        # Greedy one-to-one assignment by descending similarity
        matches: dict[str, tuple[str, str]] = {}
        used_anon: set[int] = set()
        used_prof: set[int] = set()

        # Get all (similarity, anon_idx, prof_idx) sorted descending
        flat_indices = np.argsort(sim_matrix, axis=None)[::-1]
        for flat_idx in flat_indices:
            a_idx = int(flat_idx // sim_matrix.shape[1])
            p_idx = int(flat_idx % sim_matrix.shape[1])
            sim = float(sim_matrix[a_idx, p_idx])

            if sim < threshold:
                break  # All remaining are below threshold

            if a_idx in used_anon or p_idx in used_prof:
                continue

            anon_label = anon_labels[a_idx]
            profile_name = profile_names[p_idx]
            profile_id = known_embeddings[profile_name][1]
            matches[anon_label] = (profile_name, profile_id)
            used_anon.add(a_idx)
            used_prof.add(p_idx)

        return matches

    @staticmethod
    def _is_transient_error(error_msg: str | None) -> bool:
        """Classify error as transient (retryable) or permanent."""
        if not error_msg:
            return False
        transient_patterns = [
            "timed out",
            "OOM",
            "out of memory",
            "Cannot allocate memory",
            "Daemon restarted",
            "killed",
            "signal 9",
            "signal 15",
            "connection",
            "temporary",
            "remote transcription",
            "provisioning",
            "infra",
        ]
        permanent_patterns = [
            "not found",
            "No such file",
            "not installed",
            "ImportError",
            "ModuleNotFoundError",
            "HuggingFace token is required",
            "Invalid model",
        ]
        error_lower = error_msg.lower()
        for p in permanent_patterns:
            if p.lower() in error_lower:
                return False
        for p in transient_patterns:
            if p.lower() in error_lower:
                return True
        # Default: treat unknown errors as transient (safe to retry)
        return True
