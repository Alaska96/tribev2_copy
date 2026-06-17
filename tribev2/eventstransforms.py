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
""" Train /Val splits assignments ?????????????"""
SPLIT_ATTRIBUTES = { # this dictionary is used to define the attribute name based on which splitting will be applied ,according to the study charactiristics
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
    events: pd.DataFrame, splitter: tp.Callable[str, str] # events: dataframe of events of one study
) -> pd.DataFrame:
    assert events.study.nunique() == 1, "Only one study can be assigned at a time" # make sure that one study is conisdered a time, to not mix different studies splitting , i think this also ensures fair splitting among all studies?
    study_name = events.study.unique()[0]  # double check of uniqueness of the study in the input dataframe of events
    split_by = SPLIT_ATTRIBUTES[study_name]  # each study has different attribute name that is used for splitting eg. for algonauts ="Chunck"
    events["split_attr"] = events[split_by].astype(str) # add the split attribute to the events dataframe 
    values = events["split_attr"].unique() # check the split attribute is unique to all events of same study
    # check that all rows have split attr assigned
    unassigned_event_types = events[events.split_attr.isna()].type.unique().tolist() # extractes events with unassigned split attribute
    if len(unassigned_event_types) > 0: #hanldes  unassigned events  
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
    splits = [splitter(value) for value in values]  # assign spilt name to each chunck
    if splits and "val" not in splits: # assert that at least one chunk is associated to the val set
        splits[-1] = "val"  # need at least one val split
    val_to_split = dict(zip(values, splits)) # couple chuncks with the associated split name as save in val_to_split_dict
    events["split"] = events["split_attr"].map(val_to_split) # add new column to events dataframe, named "split" and pour val_to_split_dict into it 
    return events # return the updated events dataframe which contains 2 new added columns "split_attr" :the variable based on which the spliting is made and "split": the name of assigned split

class SplitEvents(EventsTransform): # this class handles the overall splitting of all studies together 
    val_ratio: float

    def _run(self, events: pd.DataFrame) -> pd.DataFrame: # takes the dataframe of events coming from all studies and loops over the studies applying the "assign_splits" on one study a time 

        splitter = DeterministicSplitter(
            ratios={"train": 1 - self.val_ratio, "val": self.val_ratio}, seed=42
        )
        tmp = []
        for _, study_events in events.groupby("study"):
            study_events = assign_splits(study_events, splitter)
            tmp.append(study_events)
        events = pd.concat(tmp)

        return events # dataframe of events from all studies with the assigned split name


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


class RemoveDuplicates(EventsTransform):# removes redundant events from dataframe
    subset: str | tp.Sequence[str] = "filepath"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.drop_duplicates(subset=self.subset)
        return events
