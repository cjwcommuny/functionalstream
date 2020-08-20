import os
from pathlib import Path
from typing import Callable, Set, Iterable

from functionalstream.stream import Stream

LEGAL_VIDEO_FILE_EXTENSIONS: Set[str] = {
    ".mp4", ".3gp", ".ogg", ".wmv", ".webm", ".flv", ".avi", ".mkv"
}

class PathStream(Stream):
    def __init__(self, iterable: Iterable):
        super(PathStream, self).__init__(iterable)

    @staticmethod
    def from_path(path='.') -> 'PathStream':
        return PathStream(os.listdir(path))

    @staticmethod
    def from_stream(stream: Stream) -> 'PathStream':
        return PathStream(stream.iterable)

    def filter(self, function: Callable, star: bool=False) -> 'PathStream':
        return PathStream.from_stream(super().filter(function, star))

    def filterfalse(self, predicate: Callable, star: bool=False) -> 'PathStream':
        return PathStream.from_stream(super().filterfalse(predicate, star))

    def filter_dir(self) -> 'PathStream':
        return self.filter_path(os.path.isdir)

    def filter_link(self) -> 'PathStream':
        return self.filter_path(os.path.islink)

    def filter_file(self) -> 'PathStream':
        return self.filter_path(os.path.isfile)

    def filter_mount(self) -> 'PathStream':
        return self.filter_path(os.path.ismount)

    def filter_extension(self, extensions: Set[str]) -> 'PathStream':
        """
        :param extensions: e.g. {'.mp4', '.avi, '.mp3''}
        :return:
        """
        return self.filter_path(lambda path: Path(path).suffix.lower() in extensions)

    def filter_video(self) -> 'PathStream':
        return self.filter_extension(LEGAL_VIDEO_FILE_EXTENSIONS)

    def filter_path(self, func: Callable, root='') -> 'PathStream':
        return self.filter(
            lambda path: func(os.path.join(root, path))
        )

    def to_full_path(self, root='') -> 'PathStream':
        return PathStream.from_stream(self.map(lambda path: os.path.join(root, path)))

    def to_abs_path(self, root='') -> 'PathStream':
        return PathStream.from_stream(
            self.map(lambda path: os.path.abspath(os.path.join(root, path)))
        )

    def basename(self) -> 'Stream':
        return self.map(lambda path: Path(path).stem)


    def extensions(self) -> 'Stream':
        return self.map(lambda path: Path(path).suffix)

