# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import os
import typing as tp
import warnings
from pathlib import Path

import exca
import neuralset.events.etypes as ev
import pandas as pd
import torch

logger = logging.getLogger(__name__)
from neuralset.events.transforms import EventsTransform
from neuralset.events.transforms.utils import DeterministicSplitter
from tqdm import tqdm

SPLIT_ATTRIBUTES = {
    "Algonauts2025Bold": "chunk",
    "Algonauts2025": "chunk",
    "Lebel2023Bold": "task",
    "Nastase2020": "story",
    "Wen2017": "seg",
    "Wenvtwo2017": "run",
    "Lahner2024Bold": "timeline",
    "Vanessen2023": "run",
    "Aliko2020": "task",
    "Li2022": "run",
}


def assign_splits(
    events: pd.DataFrame, splitter: tp.Callable[str, str]
) -> pd.DataFrame:
    assert events.study.nunique() == 1, "Only one study can be assigned at a time"
    study_name = events.study.unique()[0]
    split_by = SPLIT_ATTRIBUTES[study_name]
    events["split_attr"] = events[split_by].astype(str)
    values = events["split_attr"].unique()
    # check that all rows have split attr assigned
    unassigned_event_types = events[events.split_attr.isna()].type.unique().tolist()
    if len(unassigned_event_types) > 0:
        msg = f"Study {study_name}: The following events do not have a split assigned and will be removed: {unassigned_event_types}"
        if any(
            [
                name.capitalize() in unassigned_event_types
                for name in ["Fmri", "Video", "Audio", "Word"]
            ]
        ):
            raise ValueError(msg)
        else:
            events = events[~events.type.isin(unassigned_event_types)]
            warnings.warn(msg)
    splits = [splitter(value) for value in values]
    if splits and "val" not in splits:
        splits[-1] = "val"  # need at least one val split
    #val_to_split = dict(zip(values, splits))
    #events["split"] = events["split_attr"].map(val_to_split)
    val_to_split = dict(zip(values, splits))
    # keep predefined test splits (e.g. Algonauts s07, no fMRI) out of train/val
    is_test = (
        events["split"].eq("test")
        if "split" in events.columns
        else pd.Series(False, index=events.index)
    )
    events["split"] = events["split_attr"].map(val_to_split)
    events.loc[is_test, "split"] = "test"
    return events


class SplitEvents(EventsTransform):
    val_ratio: float

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:

        splitter = DeterministicSplitter(
            ratios={"train": 1 - self.val_ratio, "val": self.val_ratio}, seed=42
        )
        tmp = []
        for _, study_events in events.groupby("study"):
            study_events = assign_splits(study_events, splitter)
            tmp.append(study_events)
        events = pd.concat(tmp)

        return events
class AddEpisodeContextToWords(EventsTransform):
    """Carry word context across the chunks of the same episode.

    ``AddContextToWords`` restarts the context at every timeline, i.e. at every
    chunk (Friends s01e05a / s01e05b, Movie10 bourne01 / bourne02 ...), so the
    second half of an episode starts with an empty context. This transform
    prepends the full text of all *earlier* chunks of the same episode (same
    subject, same Friends episode or same Movie10 film run) to every word's
    context. Only past words are used.

    Must run after ``AddContextToWords(max_context_len=None)``, so that the
    context of the last word of a chunk holds that whole chunk.

    Memory: all subjects watched the same clips, so every context exists once per
    subject. Identical contexts are stored once and shared between subjects,
    which keeps the (now long) context column about n_subjects times smaller.

    max_prefix_words: if set, only the last N words of the earlier chunks are
    prepended (bounds the context of long films; Friends episodes are shorter).
    """

    max_prefix_words: int | None = None

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        is_word = events.type.isin(ev.EventTypesHelper("Word").names).to_numpy()
        context = events["context"].to_numpy(dtype=object, copy=True)  # non-words keep NaN
        has_context = events["context"].fillna("").astype(bool).to_numpy()
        positions = (is_word & has_context).nonzero()[0]
        timelines = events["timeline"].to_numpy()

        # full text of each chunk = context of its last word
        words = pd.DataFrame({
            "timeline": timelines[positions],
            "start": events["start"].to_numpy()[positions],
            "pos": positions,
        })
        last = words.sort_values("start").groupby("timeline").tail(1)
        chunk_text = {tl: context[p] for tl, p in zip(last.timeline, last.pos)}
        del words, last

        # group the chunks of each timeline into episodes, in viewing order
        cols = [c for c in ("study", "subject", "task", "movie", "chunk", "run") if c in events]
        meta = events.loc[is_word, ["timeline", *cols]].groupby("timeline").first()
        keys = meta.apply(self._episode_and_order, axis=1, result_type="expand")
        meta["episode"], meta["order"] = keys[0], keys[1]

        shared: dict = {}  # one stored copy of each distinct string
        prefixes: dict[str, str] = {}
        for _, episode in meta.sort_values("order").groupby("episode", sort=False):
            past: list[str] = []
            for timeline in episode.index:
                prefix = " ".join(past)
                if self.max_prefix_words is not None:
                    prefix = " ".join(prefix.split(" ")[-self.max_prefix_words:]) if prefix else ""
                prefixes[timeline] = shared.setdefault(prefix, prefix)
                if chunk_text.get(timeline):
                    past.append(chunk_text[timeline])
        del chunk_text

        n_prefixed = 0
        for p in positions:
            prefix = prefixes.get(timelines[p], "")
            key = (prefix, context[p])
            new = shared.get(key)
            if new is None:
                new = f"{prefix} {context[p]}" if prefix else shared.setdefault(context[p], context[p])
                shared[key] = new
            context[p] = new
            n_prefixed += bool(prefix)
        del shared
        # object dtype keeps the shared strings shared (a pandas string dtype would copy them)
        events["context"] = pd.Series(context, index=events.index, dtype=object)
        print(
            f"[AddEpisodeContextToWords] episode context added to {n_prefixed} / "
            f"{len(positions)} words ({meta.episode.nunique()} episodes)"
        )
        return events

    @staticmethod
    def _episode_and_order(row: pd.Series) -> tuple[str, float]:
        chunk = str(row.get("chunk", ""))
        base = [str(row.get(c, "")) for c in ("study", "subject", "task", "movie")]
        if chunk[-1:].isalpha():  # Friends: "e05a" -> episode e05, order a < b < c
            return "/".join(base + [chunk[:-1]]), float(ord(chunk[-1]))
        # Movie10: chunks 1, 2, ... of the same film, within the same run
        return "/".join(base + [f"run-{row.get('run', '')}"]), float(chunk)


class ExtractWordsFromAudio(EventsTransform):
    """
    Language is hard-coded because auto-detection in performed on first 30s of audio, which can be empty e.g. for movies.
    """

    language: str = "english"
    overwrite: bool = False

    @staticmethod
    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        import json
        import os
        import subprocess
        import tempfile

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16"

        with tempfile.TemporaryDirectory() as output_dir:
            logger.info("Running whisperx via uvx...")
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "16",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]  # remove empty args
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"]
            sentence = sentence.replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                word_dict = {
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                }
                words.append(word_dict)

        transcript = pd.DataFrame(words)
        return transcript

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "Word" in events.type.unique():
            logger.warning("Words already present in the events dataframe, skipping")
            return events
        audio_events = events.loc[events.type == "Audio"]
        transcripts = {}
        for wav_filename in tqdm(
            audio_events.filepath.unique(),
            total=len(audio_events.filepath.unique()),
            desc="Extracting words from audio",
        ):
            wav_filename = Path(wav_filename)
            transcript_filename = wav_filename.with_suffix(".tsv")
            if transcript_filename.exists() and not self.overwrite:
                try:
                    transcript = pd.read_csv(transcript_filename, sep="\t")
                except pd.errors.EmptyDataError:
                    transcript = pd.DataFrame()
                    logger.warning(f"Empty transcript file {transcript_filename}")
            else:
                transcript = self._get_transcript_from_audio(
                    wav_filename, self.language
                )
                transcript.to_csv(transcript_filename, sep="\t", index=False)
                logger.info(f"Wrote transcript to {transcript_filename}")
            transcripts[str(wav_filename)] = transcript
        all_transcripts = []
        for audio_event in audio_events.itertuples():
            transcript = copy.deepcopy(transcripts[audio_event.filepath])
            if len(transcript) == 0:
                continue
            for k, v in audio_event._asdict().items():
                if k in (
                    "frequency",
                    "filepath",
                    "type",
                    "start",
                    "duration",
                    "offset",
                ):
                    continue
                transcript.loc[:, k] = v
            transcript["type"] = "Word"
            transcript["language"] = self.language
            transcript["start"] += audio_event.start + audio_event.offset
            all_transcripts.append(transcript)

        if all_transcripts:
            events = pd.concat([events, pd.concat(all_transcripts)], ignore_index=True)
        else:
            logger.warning("No transcripts found, skipping")
        return events


class CreateVideosFromImages(EventsTransform):
    fps: int = 10
    remove_images: bool = True
    infra: exca.MapInfra = exca.MapInfra(cluster="processpool")

    @infra.apply(
        item_uid=lambda image_event: f"{image_event.filepath}_{image_event.duration}"
    )
    def create_video(self, image_events: list[ev.Image]) -> tp.Iterator[ev.Video]:
        for image_event in image_events:
            image_filepath = Path(image_event.filepath)
            video_filepath = (
                Path(self.infra.uid_folder(create=True))
                / f"{image_filepath.stem}_{image_event.duration}.mp4"
            )
            from moviepy import ImageClip

            video_filepath.parent.mkdir(parents=True, exist_ok=True)
            clip = ImageClip(str(image_filepath), duration=image_event.duration)
            with (
                open(os.devnull, "w") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                clip.write_videofile(
                    video_filepath, codec="libx264", audio=False, fps=self.fps
                )
            video_event = ev.Video.from_dict(
                image_event.to_dict()
                | {
                    "type": "Video",
                    "filepath": str(video_filepath),
                    "frequency": self.fps,
                }
            )
            yield video_event

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        images = events.loc[events.type == "Image"]
        image_events = []
        for image in tqdm(
            images.itertuples(), total=len(images), desc="Extracting image events"
        ):
            image_events.append(ev.Image.from_dict(image._asdict()))
        video_events = [
            video_event.to_dict() for video_event in self.create_video(image_events)
        ]
        events = pd.concat([events, pd.DataFrame(video_events)], ignore_index=True)
        if self.remove_images:
            events = events.loc[events.type != "Image"]
        return events.reset_index(drop=True)


class RemoveDuplicates(EventsTransform):
    subset: str | tp.Sequence[str] = "filepath"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.drop_duplicates(subset=self.subset)
        return events
