import ast
import logging
import typing as tp
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from neuralset.events import study

logger = logging.getLogger(__name__)


class OOD_Algonauts2025(study.Study):

    "# ******************************* D6   class OOD_Algonauts2025(study.Study) was instantiated  "
    _SUBJECTS: tp.ClassVar[list[str]] = ["sub-01", "sub-02", "sub-03", "sub-05"]
    _TASKS: tp.ClassVar[list[str]] = ["ood"]
    _SPACE: tp.ClassVar[str] = "space-MNI152NLin2009cAsym"
    _ATLAS: tp.ClassVar[str] = "atlas-Schaefer18_parcel-1000Par7Net"
    _FREQUENCY: tp.ClassVar[float] = 1 / 1.49
    # study name used at training time: subject names must match the checkpoint's subject mapping
    _TRAINED_STUDY_NAME: tp.ClassVar[str] = "Algonauts2025"

    device: tp.ClassVar[str] = "Fmri"
    dataset_name: tp.ClassVar[str] = "algonauts_2025.competitors.ood"  # metadata only, not used for paths
    url: tp.ClassVar[str] = "https://algonautsproject.com/"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{algonauts2025,
        url = {https://arxiv.org/abs/2501.00504},
        author = {Gifford,  Alessandro T. and Bersch,  Domenic and St-Laurent,  Marie and Pinsard,  Basile and Boyle,  Julie and Bellec,  Lune and Oliva,  Aude and Roig,  Gemma and Cichy,  Radoslaw M.},
        keywords = {Neurons and Cognition (q-bio.NC),  FOS: Biological sciences,  FOS: Biological sciences},
        title = {The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies},
        publisher = {arXiv},
        year = {2025},
        copyright = {Creative Commons Attribution 4.0 International},
        doi={https://doi.org/10.48550/arXiv.2501.00504},
        url={https://arxiv.org/abs/2501.00504}
    }
    """
    description: tp.ClassVar[str] = (
        "Out-of-distribution (ood) subset of the Algonauts 2025 stimuli, used for inference only (no fMRI)"
    )
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "datalad>=0.19.5",
        "moviepy",
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=48,  # 4 subjects x 6 movies x 2 chunks (checked at load time)
        num_subjects=4,
        num_events_in_query=1,  # default query "timeline_index < 1" -> sub-01 chaplin1: 1 Video event
        event_types_in_query={"Video"},
    )

    def _download(self) -> None:
        raise NotImplementedError("Download method not implemented yet")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        "# ******************************* D7  class OOD_Algonauts2025(study.Study). iter_timelines(self) "
        for subject in self._SUBJECTS:
            for task in self._TASKS:
                if task == "ood":
                    movie_chunk = (
                        ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"],
                        range(1, 3),
                    )
                    for movie, chunk in product(*movie_chunk):  # type: ignore
                        tl = dict(
                            subject=subject,
                            task=task,
                            movie=movie,
                            chunk=str(chunk),
                        )
                        stim_path = self._get_movie_filepath(tl)
                        if stim_path.exists():
                            yield tl
                else:
                    raise ValueError(f"requested movie(task) is not in the inference dataset: {task}")

    def _get_transcript_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        base = (
            self.path
            / "algonauts_2025.competitors/stimuli/transcripts"
            / tl["task"]
        )
        if tl["task"] == "ood":
            if tl["movie"] == "chaplin":
                raise ValueError("chaplin has no transcript")
            return base / f"{tl['task']}/_{tl['movie']}{int(tl['chunk'])}.tsv"
        raise ValueError(f"Unknown task: {tl['task']}")

    def _get_movie_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        base = (
            self.path
            / "algonauts_2025.competitors/stimuli/movies"
            / tl["task"]
        )
        if tl["task"] == "ood":
            return base / f"task-{tl['movie']}{int(tl['chunk'])}_video.mkv"
        raise ValueError(f"Unknown task: {tl['task']}")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # ******************************* D7'   class OOD_Algonauts2025(study.Study)._load_timeline_events "
        all_events = []

        movie_filepath = self._get_movie_filepath(timeline)
        movie_event = dict(type="Video", filepath=str(movie_filepath), start=0)
        all_events.append(movie_event)

        word_events = []
        if timeline["movie"] != "chaplin":
            transcript_path = self._get_transcript_filepath(timeline)
            transcript_df = pd.read_csv(transcript_path, sep="\t")
            for _, row in transcript_df.iterrows():
                words = ast.literal_eval(row["words_per_tr"])
                starts = ast.literal_eval(row["onsets_per_tr"])
                durations = ast.literal_eval(row["durations_per_tr"])
                for word, start, duration in zip(words, starts, durations):
                    event = dict(
                        type="Word",
                        text=word,
                        start=start,
                        duration=duration,
                        stop=start + duration,
                        language="english",
                    )
                    word_events.append(event)
        if word_events:
            word_df = pd.DataFrame(word_events)
            text = " ".join(word_df["text"].tolist())
            text_event = dict(
                type="Text",
                text=text,
                start=word_df["start"].min(),
                duration=word_df["stop"].max() - word_df["start"].min(),
                stop=word_df["stop"].max(),
                language="english",
            )
            all_events.append(text_event)
        all_events.extend(word_events)

        events_df = pd.DataFrame(all_events)
        events_df["split"] = "all"

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "heard"
        )

        return events_df

    def build(self) -> pd.DataFrame:
        # neuralset names subjects "<ClassName>/<subject>"; rename to the training study's prefix
        out = super().build()
        prefix = f"{self.__class__.__name__}/"
        out["subject"] = out["subject"].str.replace(
            prefix, f"{self._TRAINED_STUDY_NAME}/", regex=False
        )
        return out
